cmake_minimum_required(VERSION 3.18)
project(cuda_secp256k1 LANGUAGES CXX CUDA)

# Pick the GPU architecture: -DCMAKE_CUDA_ARCHITECTURES=86 for an RTX 30xx, 80 for A100, ...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# The library itself is header-only.
add_library(cuda_secp256k1 INTERFACE)
target_include_directories(cuda_secp256k1 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

option(CUDA_SECP256K1_BUILD_TESTS "Build the OpenSSL-checked tests" ON)
option(CUDA_SECP256K1_BUILD_BENCH "Build the benchmark" ON)
option(CUDA_SECP256K1_BUILD_EXAMPLES "Build the examples" ON)

if(CUDA_SECP256K1_BUILD_TESTS)
    find_package(OpenSSL REQUIRED)
    enable_testing()
    foreach(name bn mont ec)
        add_executable(test_${name} tests/test_${name}.cu)
        target_include_directories(test_${name} PRIVATE tests/common)
        target_link_libraries(test_${name} PRIVATE cuda_secp256k1 OpenSSL::Crypto)
        add_test(NAME ${name} COMMAND test_${name})
    endforeach()
endif()

if(CUDA_SECP256K1_BUILD_BENCH)
    add_executable(bench_scalar_mul bench/bench_scalar_mul.cu)
    target_link_libraries(bench_scalar_mul PRIVATE cuda_secp256k1)
endif()

if(CUDA_SECP256K1_BUILD_EXAMPLES)
    add_executable(derive_pubkey examples/derive_pubkey.cu)
    target_link_libraries(derive_pubkey PRIVATE cuda_secp256k1)
endif()
