#include "runtime/core/allocator/vram_superallocator.h"
#include "runtime/core/scheduler/admission.h"
#include "runtime/kernels/decode/decode_kernels.h"
#include "runtime/loaders/q4k_manifest.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kAlignment = 256;

struct Options {
  std::filesystem::path manifest;
  std::filesystem::path tensor_plan;
  std::vector<std::string> tensors;
  std::string label = "q4k_sequence";
  int steps = 32;
  std::size_t limit = 8;
  std::size_t kv_mib = 64;
  std::size_t wddm_guard_mib = 512;
  std::uint32_t page_words = 16u * 1024u;
  std::uint32_t active_pages = 512u;
  bool chain_activation = true;
  bool source_plan_validation = true;
};

struct ResidentTensor {
  rtxllm::Q4KManifestTensor tensor;
  std::size_t payload_offset = 0;
  std::size_t metadata_offset = 0;
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

std::uint32_t parse_u32(const char* value, std::string_view name) {
  const unsigned long parsed = std::stoul(value);
  if (parsed == 0 || parsed > UINT32_MAX) {
    throw std::invalid_argument(std::string(name) + " must be in uint32 range and non-zero");
  }
  return static_cast<std::uint32_t>(parsed);
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
    if (arg == "--q4k-manifest") {
      options.manifest = require_value(arg);
    } else if (arg == "--tensor-plan") {
      options.tensor_plan = require_value(arg);
    } else if (arg == "--tensor") {
      options.tensors.push_back(require_value(arg));
    } else if (arg == "--label") {
      options.label = require_value(arg);
    } else if (arg == "--steps") {
      options.steps = static_cast<int>(parse_size(require_value(arg), arg));
    } else if (arg == "--limit") {
      options.limit = parse_size(require_value(arg), arg);
    } else if (arg == "--kv-mib") {
      options.kv_mib = parse_size(require_value(arg), arg);
    } else if (arg == "--wddm-guard-mib") {
      options.wddm_guard_mib = parse_size(require_value(arg), arg);
    } else if (arg == "--page-words") {
      options.page_words = parse_u32(require_value(arg), arg);
    } else if (arg == "--active-pages") {
      options.active_pages = parse_u32(require_value(arg), arg);
    } else if (arg == "--no-chain-activation") {
      options.chain_activation = false;
    } else if (arg == "--no-source-plan-validation") {
      options.source_plan_validation = false;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "runtime_q4k_sequence_bench --q4k-manifest PATH [--tensor NAME ...]\n"
          << "                          [--limit N] [--steps N] [--kv-mib N]\n"
          << "                          [--tensor-plan PATH]\n"
          << "                          [--no-chain-activation]\n"
          << "                          [--no-source-plan-validation]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + std::string(arg));
    }
  }
  if (options.manifest.empty()) {
    throw std::invalid_argument("--q4k-manifest is required");
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

std::string kernel_variant(const std::string& policy) {
  if (policy == "split-predecoded") {
    return "q4k_split_predecoded_vec4x4";
  }
  if (policy == "split-half") {
    return "q4k_split_half_vec4x4";
  }
  if (policy == "split-compact") {
    return "q4k_split_compact_vec4x4";
  }
  if (policy == "split-native") {
    return "q4k_split_native_vec4x4";
  }
  throw std::runtime_error("unsupported Q4_K sequence kernel policy: " + policy);
}

cudaError_t launch_tensor(
    const ResidentTensor& resident,
    const std::uint8_t* payload_base,
    const std::uint8_t* metadata_base,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream) {
  const auto* payload = payload_base + resident.payload_offset;
  const auto* metadata = metadata_base + resident.metadata_offset;
  const auto& tensor = resident.tensor;
  if (tensor.runtime_kernel == "split-predecoded") {
    return rtxllm_launch_q4k_matvec_decode_split_predecoded_vec4x4(
        payload,
        reinterpret_cast<const float*>(metadata),
        activation,
        logits,
        kv,
        token,
        tensor.rows,
        tensor.cols,
        kv_words,
        page_words,
        active_pages,
        stream);
  }
  if (tensor.runtime_kernel == "split-half") {
    return rtxllm_launch_q4k_matvec_decode_split_half_vec4x4(
        payload,
        reinterpret_cast<const std::uint16_t*>(metadata),
        activation,
        logits,
        kv,
        token,
        tensor.rows,
        tensor.cols,
        kv_words,
        page_words,
        active_pages,
        stream);
  }
  if (tensor.runtime_kernel == "split-compact") {
    return rtxllm_launch_q4k_matvec_decode_split_compact_vec4x4(
        payload,
        metadata,
        activation,
        logits,
        kv,
        token,
        tensor.rows,
        tensor.cols,
        kv_words,
        page_words,
        active_pages,
        stream);
  }
  if (tensor.runtime_kernel == "split-native") {
    return rtxllm_launch_q4k_matvec_decode_split_native_vec4x4(
        payload,
        metadata,
        activation,
        logits,
        kv,
        token,
        tensor.rows,
        tensor.cols,
        kv_words,
        page_words,
        active_pages,
        stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto manifest = rtxllm::load_q4k_manifest(options.manifest);
    const auto selected = rtxllm::select_q4k_tensors(manifest, options.tensors, options.limit);
    if (selected.empty()) {
      throw std::runtime_error("no Q4_K tensors selected");
    }
    const auto tensor_plan_path =
        options.tensor_plan.empty() ? manifest.tensor_plan : options.tensor_plan;
    bool source_plan_validated = false;
    rtxllm::Q4KTensorPlanValidation source_plan_validation;
    if (options.source_plan_validation) {
      if (tensor_plan_path.empty()) {
        throw std::runtime_error(
            "source tensor-plan validation is enabled but no tensor plan path is available");
      }
      source_plan_validation =
          rtxllm::validate_q4k_manifest_tensors_against_plan(selected, tensor_plan_path);
      source_plan_validated = true;
    }

    std::vector<ResidentTensor> resident;
    resident.reserve(selected.size());
    std::size_t payload_bytes = 0;
    std::size_t metadata_bytes = 0;
    std::uint32_t max_rows = 0;
    std::uint32_t max_cols = 0;
    std::map<std::string, std::size_t> role_counts;
    std::set<int> observed_layers;
    for (const auto& tensor : selected) {
      kernel_variant(tensor.runtime_kernel);
      resident.push_back({tensor, payload_bytes, metadata_bytes});
      payload_bytes += align_up(static_cast<std::size_t>(tensor.payload_bytes));
      metadata_bytes += align_up(static_cast<std::size_t>(tensor.metadata_bytes));
      max_rows = std::max(max_rows, tensor.rows);
      max_cols = std::max(max_cols, tensor.cols);
      role_counts[tensor.role.empty() ? "unknown" : tensor.role] += 1;
      if (tensor.layer >= 0) {
        observed_layers.insert(tensor.layer);
      }
    }

    const std::size_t activation_bytes = align_up(sizeof(float) * max_cols);
    const std::size_t logits_bytes = align_up(sizeof(float) * max_rows);
    const std::size_t workspace_bytes = activation_bytes + logits_bytes;
    const std::size_t kv_bytes = mib(options.kv_mib);
    const std::size_t dma_bytes = mib(1);
    const std::size_t resident_bytes = payload_bytes + metadata_bytes;
    const std::size_t setup_h2d_bytes =
        resident_bytes + sizeof(float) * max_cols + sizeof(std::uint32_t);

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

    const auto decision = rtxllm::decide_admission(
        rtxllm::ResidencyPolicy::Strict,
        rtxllm::MemoryBudget{free_bytes, mib(options.wddm_guard_mib)},
        rtxllm::RequestEstimate{
            resident_bytes,
            512,
            static_cast<std::size_t>(options.steps),
            1024,
            workspace_bytes + kv_bytes + dma_bytes});

    if (!decision.admit) {
      std::cout << "{\n";
      std::cout << "  \"path\": \"native_q4k_manifest_sequence_graph\",\n";
      std::cout << "  \"label\": \"" << json_escape(options.label) << "\",\n";
      std::cout << "  \"admitted\": false,\n";
      std::cout << "  \"source_plan_validated\": "
                << (source_plan_validated ? "true" : "false") << ",\n";
      if (source_plan_validated) {
        std::cout << "  \"tensor_plan\": \""
                  << json_escape(source_plan_validation.path.string()) << "\",\n";
        std::cout << "  \"source_plan_tensors_checked\": "
                  << source_plan_validation.checked_tensors << ",\n";
      }
      std::cout << "  \"reason\": \"" << json_escape(decision.reason) << "\",\n";
      std::cout << "  \"required_bytes\": " << decision.required_bytes << ",\n";
      std::cout << "  \"usable_bytes\": " << decision.usable_bytes << "\n";
      std::cout << "}\n";
      return 2;
    }

    rtxllm::VramSuperallocator pool;
    pool.initialize({
        {"weights", payload_bytes},
        {"metadata", metadata_bytes},
        {"kv_pages", kv_bytes},
        {"workspace", workspace_bytes},
        {"dma", dma_bytes},
    });

    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    pool.reset_async(stream);

    auto* payload_base = static_cast<std::uint8_t*>(pool.arena_ptr("weights"));
    auto* metadata_base = static_cast<std::uint8_t*>(pool.arena_ptr("metadata"));
    auto* workspace = static_cast<std::uint8_t*>(pool.arena_ptr("workspace"));
    auto* activation = reinterpret_cast<float*>(workspace);
    auto* logits = reinterpret_cast<float*>(workspace + activation_bytes);
    auto* kv = static_cast<float*>(pool.arena_ptr("kv_pages"));
    auto* token = static_cast<std::uint32_t*>(pool.arena_ptr("dma"));

    for (const auto& item : resident) {
      const auto payload = rtxllm::load_binary_file(
          item.tensor.payload_path, item.tensor.payload_bytes, "q4k payload");
      if (rtxllm::byte_checksum(payload) != item.tensor.payload_checksum) {
        throw std::runtime_error("payload checksum mismatch: " + item.tensor.name);
      }
      check(cudaMemcpyAsync(
                payload_base + item.payload_offset,
                payload.data(),
                payload.size(),
                cudaMemcpyHostToDevice,
                stream),
          "payload upload");

      const auto metadata = rtxllm::load_binary_file(
          item.tensor.metadata_path, item.tensor.metadata_bytes, "q4k metadata");
      if (rtxllm::byte_checksum(metadata) != item.tensor.metadata_checksum) {
        throw std::runtime_error("metadata checksum mismatch: " + item.tensor.name);
      }
      check(cudaMemcpyAsync(
                metadata_base + item.metadata_offset,
                metadata.data(),
                metadata.size(),
                cudaMemcpyHostToDevice,
                stream),
          "metadata upload");
    }

    std::vector<float> host_activation(max_cols);
    for (std::uint32_t col = 0; col < max_cols; ++col) {
      host_activation[col] = (static_cast<int>(col % 127u) - 63) / 64.0f;
    }
    const std::uint32_t initial_token = 0;
    check(cudaMemcpyAsync(
              activation,
              host_activation.data(),
              sizeof(float) * max_cols,
              cudaMemcpyHostToDevice,
              stream),
        "activation upload");
    check(cudaMemcpyAsync(token, &initial_token, sizeof(initial_token), cudaMemcpyHostToDevice, stream),
        "token upload");
    check(cudaStreamSynchronize(stream), "setup sync");

    const auto launch_sequence = [&](cudaStream_t launch_stream) {
      for (std::size_t index = 0; index < resident.size(); ++index) {
        const auto& item = resident[index];
        check(launch_tensor(
                  item,
                  payload_base,
                  metadata_base,
                  activation,
                  logits,
                  kv,
                  token,
                  static_cast<std::uint32_t>(kv_bytes / sizeof(float)),
                  options.page_words,
                  options.active_pages,
                  launch_stream),
            "q4k sequence launch");
        if (options.chain_activation) {
          const auto& next = resident[(index + 1u) % resident.size()];
          check(rtxllm_launch_activation_feedback(
                    logits,
                    activation,
                    token,
                    item.tensor.rows,
                    next.tensor.cols,
                    static_cast<std::uint32_t>(index),
                    launch_stream),
              "activation feedback");
        }
      }
    };

    std::uint32_t* host_token = nullptr;
    check(cudaHostAlloc(
              reinterpret_cast<void**>(&host_token),
              sizeof(std::uint32_t),
              cudaHostAllocDefault),
        "cudaHostAlloc");

    launch_sequence(stream);
    check(cudaMemcpyAsync(host_token, token, sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream),
        "warmup token memcpy");
    check(cudaStreamSynchronize(stream), "warmup sync");

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    const auto capture_start = std::chrono::steady_clock::now();
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
    launch_sequence(stream);
    check(cudaMemcpyAsync(host_token, token, sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream),
        "captured token memcpy");
    check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    const auto capture_stop = std::chrono::steady_clock::now();

    check(cudaGraphInstantiate(&graph_exec, graph, 0), "cudaGraphInstantiate");
    const auto upload_start = std::chrono::steady_clock::now();
    check(cudaGraphUpload(graph_exec, stream), "cudaGraphUpload");
    check(cudaStreamSynchronize(stream), "graph upload sync");
    const auto upload_stop = std::chrono::steady_clock::now();

    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<std::size_t>(options.steps));
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < options.steps; ++step) {
      const auto step_start = std::chrono::steady_clock::now();
      check(cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch");
      check(cudaStreamSynchronize(stream), "graph replay sync");
      const auto step_stop = std::chrono::steady_clock::now();
      latencies_ms.push_back(
          std::chrono::duration<double, std::milli>(step_stop - step_start).count());
    }
    const auto stop = std::chrono::steady_clock::now();

    const double wall_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    const double capture_ms =
        std::chrono::duration<double, std::milli>(capture_stop - capture_start).count();
    const double upload_ms =
        std::chrono::duration<double, std::milli>(upload_stop - upload_start).count();
    const double mean_ms =
        std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / latencies_ms.size();
    const auto last_token = *host_token;

    check(cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy");
    check(cudaGraphDestroy(graph), "cudaGraphDestroy");
    check(cudaFreeHost(host_token), "cudaFreeHost");
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");

    std::cout << "{\n";
    std::cout << "  \"path\": \"native_q4k_manifest_sequence_graph\",\n";
    std::cout << "  \"label\": \"" << json_escape(options.label) << "\",\n";
    std::cout << "  \"admitted\": true,\n";
    std::cout << "  \"manifest\": \"" << json_escape(options.manifest.string()) << "\",\n";
    std::cout << "  \"tensor_count\": " << resident.size() << ",\n";
    std::cout << "  \"source_plan_validated\": "
              << (source_plan_validated ? "true" : "false") << ",\n";
    if (source_plan_validated) {
      std::cout << "  \"tensor_plan\": \"" << json_escape(source_plan_validation.path.string())
                << "\",\n";
      std::cout << "  \"source_plan_tensors_checked\": "
                << source_plan_validation.checked_tensors << ",\n";
    }
    std::cout << "  \"steps\": " << options.steps << ",\n";
    std::cout << "  \"chain_activation\": " << (options.chain_activation ? "true" : "false") << ",\n";
    std::cout << "  \"activation_feedbacks_per_graph\": "
              << (options.chain_activation ? resident.size() : 0u) << ",\n";
    std::cout << "  \"kernel_launches_per_graph\": "
              << (resident.size() + (options.chain_activation ? resident.size() : 0u)) << ",\n";
    std::cout << "  \"graph_replay_rate\": 1.0,\n";
    std::cout << "  \"wall_ms\": " << wall_ms << ",\n";
    std::cout << "  \"steps_per_second\": " << (options.steps * 1000.0 / wall_ms) << ",\n";
    std::cout << "  \"tensor_launches_per_second\": "
              << (resident.size() * options.steps * 1000.0 / wall_ms) << ",\n";
    std::cout << "  \"graph_kernel_launches_per_second\": "
              << ((resident.size() + (options.chain_activation ? resident.size() : 0u)) *
                  options.steps * 1000.0 / wall_ms) << ",\n";
    std::cout << "  \"p50_ms\": " << percentile(latencies_ms, 50) << ",\n";
    std::cout << "  \"p95_ms\": " << percentile(latencies_ms, 95) << ",\n";
    std::cout << "  \"p99_ms\": " << percentile(latencies_ms, 99) << ",\n";
    std::cout << "  \"latency_mean_ms\": " << mean_ms << ",\n";
    std::cout << "  \"capture_ms\": " << capture_ms << ",\n";
    std::cout << "  \"graph_upload_ms\": " << upload_ms << ",\n";
    std::cout << "  \"payload_bytes\": " << payload_bytes << ",\n";
    std::cout << "  \"metadata_bytes\": " << metadata_bytes << ",\n";
    std::cout << "  \"resident_bytes\": " << resident_bytes << ",\n";
    std::cout << "  \"allocated_bytes\": " << pool.total_bytes() << ",\n";
    std::cout << "  \"setup_h2d_bytes\": " << setup_h2d_bytes << ",\n";
    std::cout << "  \"h2d_bytes\": 0,\n";
    std::cout << "  \"d2h_bytes\": " << (sizeof(std::uint32_t) * options.steps) << ",\n";
    std::cout << "  \"max_rows\": " << max_rows << ",\n";
    std::cout << "  \"max_cols\": " << max_cols << ",\n";
    std::cout << "  \"layer_count_observed\": " << observed_layers.size() << ",\n";
    std::cout << "  \"last_token\": " << last_token << ",\n";
    std::cout << "  \"role_counts\": {";
    std::size_t role_index = 0;
    for (const auto& [role, count] : role_counts) {
      if (role_index++ != 0) {
        std::cout << ", ";
      }
      std::cout << "\"" << json_escape(role) << "\": " << count;
    }
    std::cout << "},\n";
    std::cout << "  \"tensors\": [";
    for (std::size_t index = 0; index < resident.size(); ++index) {
      if (index != 0) {
        std::cout << ", ";
      }
      std::cout << "\"" << json_escape(resident[index].tensor.name) << "\"";
    }
    std::cout << "]\n";
    std::cout << "}\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }
}
