cmake_minimum_required(VERSION 3.20)
project(VortexGPUAudioBackend VERSION 1.0.0 LANGUAGES CXX)

# Set C++20/23 standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(MSVC)
    add_compile_options(/std:c++23 /permissive-)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# Find CUDA (for GPU acceleration)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    set(CUDA_SEPARABLE_COMPILATION ON)
else()
    message(WARNING "CUDA not found - GPU acceleration will be disabled")
endif()

# Find OpenCL (for cross-platform GPU support)
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    message(STATUS "OpenCL found")
else()
    message(WARNING "OpenCL not found - cross-platform GPU support will be limited")
endif()

# Find Vulkan (for modern GPU compute)
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    message(STATUS "Vulkan found")
else()
    message(WARNING "Vulkan not found - modern GPU compute will be disabled")
endif()

# Audio libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(JUCE REQUIRED juce)
pkg_check_modules(SNDFILE REQUIRED sndfile)
pkg_check_modules(FLAC REQUIRED flac)
pkg_check_modules(VORBIS REQUIRED vorbisfile vorbis)
pkg_check_modules(MP3LAME REQUIRED lame)

# GPU-accelerated libraries
if(CUDA_FOUND)
    find_package(CUBLAS)
    find_package(CUFFT)
endif()

# Intel IPP for optimized DSP
find_package(IPP QUIET)
if(IPP_FOUND)
    message(STATUS "Intel IPP found - enabling optimized DSP")
    set(IPP_LIBRARIES ippcore ippvm ipp s ippcc)
endif()

# FFTW for convolution processing
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3f)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${JUCE_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIRS}
)

# Source files
set(CORE_SOURCES
    src/core/audio_engine.cpp
    src/core/audio_processor.cpp
    src/core/dsp/dsd_processor.cpp
    src/core/dsp/eq_processor.cpp
    src/core/dsp/convolver.cpp
    src/core/dsp/resampler.cpp
    src/core/filters/biquad_filter.cpp
    src/core/filters/fir_filter.cpp
    src/core/filters/filter_chain.cpp
    src/core/fileio/audio_file_loader.cpp
    src/core/fileio/format_detector.cpp
    src/core/fileio/metadata_extractor.cpp
    src/core/processing_chain.cpp
    src/core/audio_buffer_manager.cpp
    src/core/progress_tracker.cpp
)

set(GPU_SOURCES
    src/core/gpu/gpu_processor.cpp
    src/core/gpu/memory_manager.cpp
)

set(NETWORK_SOURCES
    src/network/http_server.cpp
    src/network/websocket_server.cpp
    src/network/discovery_service.cpp
    src/network/protocol/binary_protocol.cpp
)

set(OUTPUT_SOURCES
    src/output/output_manager.cpp
    src/output/local_output.cpp
    src/output/roon_bridge.cpp
    src/output/hqplayer_naa.cpp
    src/output/upnp_renderer.cpp
)

set(SYSTEM_SOURCES
    src/system/hardware_monitor.cpp
    src/system/latency_analyzer.cpp
    src/system/performance_counter.cpp
)

set(UTILS_SOURCES
    src/utils/logger.cpp
    src/utils/config_manager.cpp
    src/utils/thread_pool.cpp
)

# GPU processor sources
set(CUDA_SOURCES "")
set(OPENCL_SOURCES "")
set(VULKAN_SOURCES "")

if(CUDA_FOUND)
    list(APPEND CUDA_SOURCES
        src/core/gpu/cuda_processor.cpp
    )
endif()

if(OpenCL_FOUND)
    list(APPEND OPENCL_SOURCES
        src/core/gpu/opencl_processor.cpp
    )
endif()

if(Vulkan_FOUND)
    list(APPEND VULKAN_SOURCES
        src/core/gpu/vulkan_processor.cpp
    )
endif()

# Create library for audio core
add_library(vortex_audio_core STATIC
    ${CORE_SOURCES}
    ${GPU_SOURCES}
    ${CUDA_SOURCES}
    ${OPENCL_SOURCES}
    ${VULKAN_SOURCES}
    ${SYSTEM_SOURCES}
    ${UTILS_SOURCES}
)

target_include_directories(vortex_audio_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(vortex_audio_core PUBLIC
    Threads::Threads
    ${JUCE_LIBRARIES}
    ${SNDFILE_LIBRARIES}
    ${FLAC_LIBRARIES}
    ${VORBIS_LIBRARIES}
    ${MP3LAME_LIBRARIES}
    ${FFTW_LIBRARIES}
)

# Link GPU libraries
if(CUDA_FOUND)
    target_link_libraries(vortex_audio_core PUBLIC
        ${CUDA_LIBRARIES}
        CUDA::cudart
        CUDA::cublas
        CUDA::cufft
    )
endif()

if(OpenCL_FOUND)
    target_link_libraries(vortex_audio_core PUBLIC
        ${OpenCL_LIBRARIES}
    )
endif()

if(Vulkan_FOUND)
    target_link_libraries(vortex_audio_core PUBLIC
        Vulkan::Vulkan
    )
endif()

# Link Intel IPP if available
if(IPP_FOUND)
    target_link_libraries(vortex_audio_core PUBLIC
        ${IPP_LIBRARIES}
    )
endif()

# Network services library (Rust integration)
add_library(vortex_network STATIC
    ${NETWORK_SOURCES}
)

target_link_libraries(vortex_network PUBLIC
    vortex_audio_core
    Threads::Threads
)

# Output device management library
add_library(vortex_output STATIC
    ${OUTPUT_SOURCES}
)

target_link_libraries(vortex_output PUBLIC
    vortex_audio_core
    vortex_network
    Threads::Threads
)

# Main executable
add_executable(vortex-backend
    src/main.cpp
)

target_link_libraries(vortex-backend PRIVATE
    vortex_audio_core
    vortex_network
    vortex_output
)

# Rust integration
find_program(CARGO REQUIRED)
execute_process(
    COMMAND ${CARGO} metadata --format-version 1 --no-deps
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE CARGO_METADATA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(CARGO_METADATA)
    # Parse Cargo.toml to get target directory
    execute_process(
        COMMAND ${CARGO} metadata --format-version 1 --no-deps --message-format=plain
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE CARGO_TARGET_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    string(FIND "target directory: " CARGO_TARGET_DIR POS)
    string(SUBSTRING ${CARGO_TARGET_DIR} ${POS} -1 CARGO_TARGET_DIR)

    # Create Rust static library
    execute_process(
        COMMAND ${CARGO} build --release --target-dir ${CMAKE_BINARY_DIR}/rust
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Add Rust library as prebuilt
    add_library(vortex_rust STATIC IMPORTED)
    set_property(TARGET vortex_rust IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/rust/liblibvortex_network.a)

    target_include_directories(vortex_rust INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/network/rust/target/release/deps
    )

    target_link_libraries(vortex_audio_core PRIVATE vortex_rust)
    target_link_libraries(vortex_network PRIVATE vortex_rust)
    target_link_libraries(vortex_output PRIVATE vortex_rust)

    # Link Rust-specific libraries
    target_link_libraries(vortex_rust INTERFACE
        pthread
        dl
    )
endif()

# Install targets
install(TARGETS vortex-backend
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

# Tests
enable_testing()

# GoogleTest
find_package(GTest REQUIRED)
find_package(GMock REQUIRED)

# Test sources
set(TEST_SOURCES
    tests/unit/test_audio_engine.cpp
    tests/unit/test_gpu_processor.cpp
    tests/integration/test_websocket_protocol.cpp
    tests/performance/test_gpu_memory.cpp
    tests/unit/test_format_detector.cpp
    tests/unit/test_spectrum_analyzer.cpp
    tests/unit/test_equalizer.cpp
    tests/performance/test_convolution.cpp
    tests/gpu/test_filter_chain_gpu.cpp
    tests/contract/test_audio_upload.cpp
    tests/contract/test_websocket_realtime.cpp
    tests/contract/test_processing_chain.cpp
    tests/integration/test_dsd_processing.cpp
    tests/integration/test_roon_bridge.cpp
    tests/integration/test_hqplayer_naa.cpp
    tests/integration/test_upnp_renderer.cpp
    tests/performance/test_file_processing.cpp
    tests/performance/test_visualization_fps.cpp
    tests/e2e/test_full_pipeline.cpp
    tests/stress/test_24hour_stability.cpp
    tests/integration/test_concurrent_users.cpp
)

# Test executable
add_executable(vortex_tests
    ${TEST_SOURCES}
)

target_link_libraries(vortex_tests PRIVATE
    vortex_audio_core
    vortex_network
    vortex_output
    GTest::gtest
    GTest::gmock
    Threads::Threads
)

include(GoogleTest)
gtest_discover_tests(vortex_tests)

# Custom targets
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS vortex_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# GPU shaders compilation
if(CUDA_FOUND)
    set(SHADER_SOURCES
        shaders/audio_processing.comp
        shaders/spectrum_analyzer.comp
        shaders/convolution.comp
    )

    foreach(SHADER ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
        add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/${SHADER_NAME}.ptx
            COMMAND ${CUDA_NVCC_EXECUTABLE}
                -ptx
                -arch=${CUDA_ARCHITECTURES}
                ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
                -o ${CMAKE_BINARY_DIR}/${SHADER_NAME}.ptx
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
            VERBATIM
        )
        list(APPEND PTX_FILES ${CMAKE_BINARY_DIR}/${SHADER_NAME}.ptx)
    endforeach()

    add_custom_target(gpu_shaders ALL DEPENDS ${PTX_FILES})
endif()

# Configuration files
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config/default.json.in
    ${CMAKE_CURRENT_BINARY_DIR}/config/default.json
    @ONLY
)

# Package configuration
set(CPACK_PACKAGE_NAME "vortex-gpu-audio-backend")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION "High-performance GPU-accelerated audio processing backend")
set(CPACK_PACKAGE_VENDOR "Vortex Audio")
set(CPACK_GENERATOR "TGZ;DEB")

include(CPack)