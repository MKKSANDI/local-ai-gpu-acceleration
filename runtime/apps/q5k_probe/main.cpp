#include "runtime/core/allocator/vram_superallocator.h"
#include "runtime/core/scheduler/admission.h"
#include "runtime/kernels/decode/decode_kernels.h"
#include "runtime/loaders/iq_manifest.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kAlignment = 256;
constexpr std::uint64_t kQ5KBlockBytes = 176;
constexpr std::uint64_t kQ5KDOffset = 0;
constexpr std::uint64_t kQ5KDMinOffset = 2;
constexpr std::uint64_t kQ5KScalesOffset = 4;
constexpr std::uint64_t kQ5KScalesBytes = 12;
constexpr std::uint64_t kQ5KQhOffset = kQ5KScalesOffset + kQ5KScalesBytes;
constexpr std::uint64_t kQ5KQhBytes = 32;
constexpr std::uint64_t kQ5KQsOffset = kQ5KQhOffset + kQ5KQhBytes;
constexpr std::uint64_t kQ5KQsBytes = 128;
constexpr std::size_t kOutputWords = 7;

struct Options {
  std::filesystem::path manifest;
  std::string tensor;
  std::string label = "q5k_probe";
  int steps = 8;
  std::size_t wddm_guard_mib = 512;
};

struct Reference {
  std::uint64_t payload_checksum = 0;
  std::uint64_t first_d_raw = 0;
  std::uint64_t first_dmin_raw = 0;
  std::uint64_t first_scales_sum = 0;
  std::uint64_t first_qh_sum = 0;
  std::uint64_t first_qs_sum = 0;
  std::uint64_t block_count = 0;
};

std::size_t mib(std::size_t value) {
  return value * 1024ull * 1024ull;
}

std::size_t align_up(std::size_t value) {
  const auto mask = kAlignment - 1;
  return (value + mask) & ~mask;
}

void check(cudaError_t status, const char* op) {
  if (status != cudaSuccess) {
    std::cerr << op << " failed: " << cudaGetErrorString(status) << "\n";
    std::exit(1);
  }
}

double percentile(std::vector<double> values, double pct) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(
      std::min<double>(values.size() - 1, std::ceil((pct / 100.0) * values.size()) - 1));
  return values[index];
}

std::size_t parse_size(const char* value, std::string_view name) {
  const unsigned long long parsed = std::stoull(value);
  if (parsed == 0) {
    throw std::invalid_argument(std::string(name) + " must be non-zero");
  }
  return static_cast<std::size_t>(parsed);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const auto require_value = [&](std::string_view name) -> const char* {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string(name) + " requires a value");
      }
      return argv[++i];
    };
    if (arg == "--q5k-manifest") {
      options.manifest = require_value(arg);
    } else if (arg == "--tensor") {
      options.tensor = require_value(arg);
    } else if (arg == "--label") {
      options.label = require_value(arg);
    } else if (arg == "--steps") {
      options.steps = static_cast<int>(parse_size(require_value(arg), arg));
    } else if (arg == "--wddm-guard-mib") {
      options.wddm_guard_mib = parse_size(require_value(arg), arg);
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "runtime_q5k_probe --q5k-manifest PATH [--tensor NAME]\n"
          << "                   [--label LABEL] [--steps N]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + std::string(arg));
    }
  }
  if (options.manifest.empty()) {
    throw std::invalid_argument("--q5k-manifest is required");
  }
  return options;
}

std::string json_escape(std::string_view value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += ch;
        break;
    }
  }
  return out;
}

std::vector<std::uint8_t> read_binary_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open binary file: " + path.string());
  }
  in.seekg(0, std::ios::end);
  const auto size = static_cast<std::size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> data(size);
  if (size > 0) {
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!in) {
      throw std::runtime_error("failed to read binary file: " + path.string());
    }
  }
  return data;
}

std::uint64_t byte_checksum(const std::vector<std::uint8_t>& data) {
  return std::accumulate(
      data.begin(),
      data.end(),
      std::uint64_t{0},
      [](std::uint64_t acc, std::uint8_t value) {
        return acc + static_cast<std::uint64_t>(value);
      });
}

std::uint64_t range_sum(
    const std::vector<std::uint8_t>& data,
    std::size_t offset,
    std::size_t bytes) {
  return std::accumulate(
      data.begin() + static_cast<std::ptrdiff_t>(offset),
      data.begin() + static_cast<std::ptrdiff_t>(offset + bytes),
      std::uint64_t{0},
      [](std::uint64_t acc, std::uint8_t value) {
        return acc + static_cast<std::uint64_t>(value);
      });
}

Reference make_reference(
    const std::vector<std::uint8_t>& payload,
    const rtxllm::IQManifestTensor& tensor) {
  if (tensor.block_count == 0 || tensor.payload_bytes != payload.size()) {
    throw std::runtime_error("Q5_K reference payload size mismatch");
  }
  if (payload.size() < kQ5KBlockBytes) {
    throw std::runtime_error("Q5_K payload is smaller than one block");
  }
  Reference reference;
  reference.payload_checksum = byte_checksum(payload);
  reference.first_d_raw =
      static_cast<std::uint64_t>(payload[kQ5KDOffset]) |
      (static_cast<std::uint64_t>(payload[kQ5KDOffset + 1]) << 8u);
  reference.first_dmin_raw =
      static_cast<std::uint64_t>(payload[kQ5KDMinOffset]) |
      (static_cast<std::uint64_t>(payload[kQ5KDMinOffset + 1]) << 8u);
  reference.first_scales_sum = range_sum(payload, kQ5KScalesOffset, kQ5KScalesBytes);
  reference.first_qh_sum = range_sum(payload, kQ5KQhOffset, kQ5KQhBytes);
  reference.first_qs_sum = range_sum(payload, kQ5KQsOffset, kQ5KQsBytes);
  reference.block_count = tensor.block_count;
  return reference;
}

bool reference_passed(
    const Reference& reference,
    const std::uint64_t* observed) {
  return observed[0] == reference.payload_checksum &&
         observed[1] == reference.first_d_raw &&
         observed[2] == reference.first_dmin_raw &&
         observed[3] == reference.first_scales_sum &&
         observed[4] == reference.first_qh_sum &&
         observed[5] == reference.first_qs_sum &&
         observed[6] == reference.block_count;
}

void print_rejection(
    const Options& options,
    const rtxllm::IQManifestTensor& tensor,
    const rtxllm::AdmissionDecision& decision) {
  std::cout << "{\n";
  std::cout << "  \"path\": \"native_q5k_probe\",\n";
  std::cout << "  \"label\": \"" << json_escape(options.label) << "\",\n";
  std::cout << "  \"admitted\": false,\n";
  std::cout << "  \"manifest\": \"" << json_escape(options.manifest.string()) << "\",\n";
  std::cout << "  \"tensor\": \"" << json_escape(tensor.name) << "\",\n";
  std::cout << "  \"type\": \"" << json_escape(tensor.type) << "\",\n";
  std::cout << "  \"reason\": \"" << json_escape(decision.reason) << "\",\n";
  std::cout << "  \"required_bytes\": " << decision.required_bytes << ",\n";
  std::cout << "  \"usable_bytes\": " << decision.usable_bytes << "\n";
  std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto manifest = rtxllm::load_iq_manifest(options.manifest);
    const auto tensor = rtxllm::select_iq_tensor(manifest, options.tensor);
    if (tensor.type != "Q5_K") {
      throw std::runtime_error("runtime_q5k_probe supports Q5_K only");
    }
    if (tensor.block_type_size != kQ5KBlockBytes) {
      throw std::runtime_error("Q5_K block type size mismatch");
    }

    const std::filesystem::path payload_path = tensor.payload_path.is_absolute()
        ? tensor.payload_path
        : options.manifest.parent_path() / tensor.payload_path;
    const auto payload = read_binary_file(payload_path);
    if (payload.size() != tensor.payload_bytes) {
      throw std::runtime_error(
          "Q5_K payload file size does not match manifest: " + payload_path.string());
    }
    const auto reference = make_reference(payload, tensor);
    if (reference.payload_checksum != tensor.payload_checksum ||
        reference.payload_checksum != tensor.source_checksum) {
      throw std::runtime_error("Q5_K payload checksum does not match manifest: " + payload_path.string());
    }

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    const auto output_bytes = align_up(kOutputWords * sizeof(std::uint64_t));
    const auto decision = rtxllm::decide_admission(
        rtxllm::ResidencyPolicy::Strict,
        rtxllm::MemoryBudget{free_bytes, mib(options.wddm_guard_mib)},
        rtxllm::RequestEstimate{
            align_up(payload.size()),
            1,
            static_cast<std::size_t>(options.steps),
            0,
            mib(1) + output_bytes});
    if (!decision.admit) {
      print_rejection(options, tensor, decision);
      return 2;
    }

    rtxllm::VramSuperallocator pool;
    pool.initialize({
        {"weights", align_up(payload.size())},
        {"workspace", mib(1)},
        {"dma", output_bytes},
    });

    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    auto* device_payload = static_cast<std::uint8_t*>(pool.arena_ptr("weights"));
    auto* device_output = static_cast<std::uint64_t*>(pool.arena_ptr("dma"));
    check(cudaMemcpyAsync(
              device_payload,
              payload.data(),
              payload.size(),
              cudaMemcpyHostToDevice,
              stream),
          "cudaMemcpyAsync(payload)");

    std::uint64_t* host_output = nullptr;
    check(cudaHostAlloc(
              reinterpret_cast<void**>(&host_output),
              kOutputWords * sizeof(std::uint64_t),
              cudaHostAllocDefault),
          "cudaHostAlloc(output)");

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    const auto capture_start = std::chrono::steady_clock::now();
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
    check(cudaMemsetAsync(device_output, 0, kOutputWords * sizeof(std::uint64_t), stream), "cudaMemsetAsync(output)");
    check(rtxllm_launch_q5k_probe(device_payload, tensor.block_count, device_output, stream), "q5k_probe");
    check(cudaMemcpyAsync(
              host_output,
              device_output,
              kOutputWords * sizeof(std::uint64_t),
              cudaMemcpyDeviceToHost,
              stream),
          "cudaMemcpyAsync(output)");
    check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    const auto capture_stop = std::chrono::steady_clock::now();
    check(cudaGraphInstantiate(&graph_exec, graph, 0), "cudaGraphInstantiate");
    const auto upload_start = std::chrono::steady_clock::now();
    check(cudaGraphUpload(graph_exec, stream), "cudaGraphUpload");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(upload)");
    const auto upload_stop = std::chrono::steady_clock::now();

    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<std::size_t>(options.steps));
    const auto wall_start = std::chrono::steady_clock::now();
    for (int step = 0; step < options.steps; ++step) {
      const auto start = std::chrono::steady_clock::now();
      check(cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch");
      check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(step)");
      const auto stop = std::chrono::steady_clock::now();
      latencies_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    const auto wall_stop = std::chrono::steady_clock::now();

    const bool passed = reference_passed(reference, host_output);
    const double wall_ms =
        std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
    const double mean_ms = latencies_ms.empty()
        ? 0.0
        : std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) /
              static_cast<double>(latencies_ms.size());
    const double capture_ms =
        std::chrono::duration<double, std::milli>(capture_stop - capture_start).count();
    const double upload_ms =
        std::chrono::duration<double, std::milli>(upload_stop - upload_start).count();

    std::cout << "{\n";
    std::cout << "  \"path\": \"native_q5k_probe\",\n";
    std::cout << "  \"label\": \"" << json_escape(options.label) << "\",\n";
    std::cout << "  \"admitted\": true,\n";
    std::cout << "  \"reference_passed\": " << (passed ? "true" : "false") << ",\n";
    std::cout << "  \"manifest\": \"" << json_escape(options.manifest.string()) << "\",\n";
    std::cout << "  \"tensor\": \"" << json_escape(tensor.name) << "\",\n";
    std::cout << "  \"type\": \"" << json_escape(tensor.type) << "\",\n";
    std::cout << "  \"runtime_kernel\": \"" << json_escape(tensor.runtime_kernel) << "\",\n";
    std::cout << "  \"block_count\": " << tensor.block_count << ",\n";
    std::cout << "  \"block_type_size\": " << tensor.block_type_size << ",\n";
    std::cout << "  \"payload_bytes\": " << payload.size() << ",\n";
    std::cout << "  \"resident_bytes\": " << tensor.resident_bytes << ",\n";
    std::cout << "  \"host_weight_checksum\": " << reference.payload_checksum << ",\n";
    std::cout << "  \"gpu_payload_checksum\": " << host_output[0] << ",\n";
    std::cout << "  \"first_d_raw_expected\": " << reference.first_d_raw << ",\n";
    std::cout << "  \"first_d_raw_observed\": " << host_output[1] << ",\n";
    std::cout << "  \"first_dmin_raw_expected\": " << reference.first_dmin_raw << ",\n";
    std::cout << "  \"first_dmin_raw_observed\": " << host_output[2] << ",\n";
    std::cout << "  \"first_scales_sum_expected\": " << reference.first_scales_sum << ",\n";
    std::cout << "  \"first_scales_sum_observed\": " << host_output[3] << ",\n";
    std::cout << "  \"first_qh_sum_expected\": " << reference.first_qh_sum << ",\n";
    std::cout << "  \"first_qh_sum_observed\": " << host_output[4] << ",\n";
    std::cout << "  \"first_qs_sum_expected\": " << reference.first_qs_sum << ",\n";
    std::cout << "  \"first_qs_sum_observed\": " << host_output[5] << ",\n";
    std::cout << "  \"block_count_observed\": " << host_output[6] << ",\n";
    std::cout << "  \"steps\": " << options.steps << ",\n";
    std::cout << "  \"wall_ms\": " << wall_ms << ",\n";
    std::cout << "  \"steps_per_second\": " << (options.steps * 1000.0 / wall_ms) << ",\n";
    std::cout << "  \"p50_ms\": " << percentile(latencies_ms, 50) << ",\n";
    std::cout << "  \"p95_ms\": " << percentile(latencies_ms, 95) << ",\n";
    std::cout << "  \"p99_ms\": " << percentile(latencies_ms, 99) << ",\n";
    std::cout << "  \"latency_mean_ms\": " << mean_ms << ",\n";
    std::cout << "  \"capture_ms\": " << capture_ms << ",\n";
    std::cout << "  \"graph_upload_ms\": " << upload_ms << ",\n";
    std::cout << "  \"graph_replay_rate\": 1.0,\n";
    std::cout << "  \"allocated_bytes\": " << pool.total_bytes() << ",\n";
    std::cout << "  \"setup_h2d_bytes\": " << payload.size() << ",\n";
    std::cout << "  \"h2d_bytes\": 0,\n";
    std::cout << "  \"d2h_bytes\": " << (kOutputWords * sizeof(std::uint64_t) * options.steps) << ",\n";
    std::cout << "  \"weight_source\": \"q5k_manifest_payload_file\"\n";
    std::cout << "}\n";

    check(cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy");
    check(cudaGraphDestroy(graph), "cudaGraphDestroy");
    check(cudaFreeHost(host_output), "cudaFreeHost(output)");
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return passed ? 0 : 3;
  } catch (const std::exception& exc) {
    std::cerr << "runtime_q5k_probe failed: " << exc.what() << "\n";
    return 1;
  }
}
