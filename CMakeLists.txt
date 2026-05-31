cmake_minimum_required(VERSION 3.27)

project(rtx3060_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(RTXLLM_ENABLE_CUDA "Build CUDA runtime pieces" ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(rtxllm_core
  runtime/core/allocator/vram_superallocator.cpp
  runtime/core/scheduler/admission.cpp
  runtime/core/scheduler/layer_plan.cpp
  runtime/loaders/iq_manifest.cpp
  runtime/loaders/q4k_manifest.cpp
)

target_include_directories(rtxllm_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(rtxllm_core
  PUBLIC
    RTXLLM_ENABLE_CUDA=1
    NOMINMAX
)

target_link_libraries(rtxllm_core
  PUBLIC
    CUDA::cudart
)

add_executable(runtime_layer_plan_selftest
  runtime/apps/layer_plan_selftest/main.cpp
)

target_link_libraries(runtime_layer_plan_selftest
  PRIVATE
    rtxllm_core
)

add_library(rtxllm_kernels
  runtime/kernels/decode/decode_kernels.cu
)

target_include_directories(rtxllm_kernels
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(rtxllm_kernels
  PUBLIC
    CUDA::cudart
)

set_target_properties(rtxllm_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_ARCHITECTURES 86
)

add_executable(runtime_bench
  runtime/apps/benchmark_runner/main.cpp
)

target_link_libraries(runtime_bench
  PRIVATE
    rtxllm_core
    rtxllm_kernels
    CUDA::cudart
)

add_executable(runtime_graph_bench
  runtime/apps/graph_benchmark_runner/main.cpp
)

target_link_libraries(runtime_graph_bench
  PRIVATE
    rtxllm_core
    rtxllm_kernels
    CUDA::cudart
)

add_executable(runtime_fused_bench
  runtime/apps/fused_benchmark_runner/main.cpp
)

target_link_libraries(runtime_fused_bench
  PRIVATE
    rtxllm_core
    rtxllm_kernels
    CUDA::cudart
)

add_executable(runtime_q4k_sequence_bench
  runtime/apps/q4k_sequence_benchmark_runner/main.cpp
)

target_link_libraries(runtime_q4k_sequence_bench
  PRIVATE
    rtxllm_core
    rtxllm_kernels
    CUDA::cudart
)

add_executable(runtime_iq_probe
  runtime/apps/iq_probe/main.cpp
)

target_link_libraries(runtime_iq_probe
  PRIVATE
    rtxllm_core
    rtxllm_kernels
    CUDA::cudart
)

add_executable(runtime_q5k_probe
  runtime/apps/q5k_probe/main.cpp
)

target_link_libraries(runtime_q5k_probe
  PRIVATE
    rtxllm_core
    rtxllm_kernels
    CUDA::cudart
)

add_executable(runtime_q5k_matvec_probe
  runtime/apps/q5k_matvec_probe/main.cpp
)

target_link_libraries(runtime_q5k_matvec_probe
  PRIVATE
    rtxllm_core
    rtxllm_kernels
    CUDA::cudart
)

add_executable(runtime_iq_matvec_probe
  runtime/apps/iq_matvec_probe/main.cpp
)

target_link_libraries(runtime_iq_matvec_probe
  PRIVATE
    rtxllm_core
    rtxllm_kernels
    CUDA::cudart
)

add_executable(runtime_iq_sequence_bench
  runtime/apps/iq_sequence_benchmark_runner/main.cpp
)

target_link_libraries(runtime_iq_sequence_bench
  PRIVATE
    rtxllm_core
    rtxllm_kernels
    CUDA::cudart
)

add_executable(runtime_mixed_sequence_bench
  runtime/apps/mixed_sequence_benchmark_runner/main.cpp
)

target_link_libraries(runtime_mixed_sequence_bench
  PRIVATE
    rtxllm_core
    rtxllm_kernels
    CUDA::cudart
)

add_executable(runtime_gguf_probe
  runtime/apps/gguf_probe/main.cpp
)

target_include_directories(runtime_gguf_probe
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
