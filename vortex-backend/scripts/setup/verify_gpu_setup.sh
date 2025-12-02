#!/bin/bash

# GPU Setup Verification Script
# This script verifies that the GPU development environment is properly configured

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Vortex GPU Audio Backend - GPU Setup Verification${NC}"
echo "================================================"

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run test and record result
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_exit_code="${3:-0}"

    echo -n "Testing $test_name... "

    if eval "$test_command" >/dev/null 2>&1; then
        local exit_code=$?
        if [ $exit_code -eq $expected_exit_code ]; then
            echo -e "${GREEN}PASS${NC}"
            ((TESTS_PASSED++))
            return 0
        else
            echo -e "${RED}FAIL (exit code $exit_code)${NC}"
            ((TESTS_FAILED++))
            return 1
        fi
    else
        echo -e "${RED}FAIL${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Function to check file exists
check_file() {
    local file="$1"
    local description="$2"

    echo -n "Checking $description... "

    if [ -f "$file" ]; then
        echo -e "${GREEN}FOUND${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}NOT FOUND${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Function to check command exists
check_command() {
    local cmd="$1"
    local description="$2"

    echo -n "Checking $description... "

    if command -v "$cmd" >/dev/null 2>&1; then
        echo -e "${GREEN}AVAILABLE${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}NOT AVAILABLE${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Function to check library exists
check_library() {
    local lib="$1"
    local description="$2"

    echo -n "Checking $description... "

    if ldconfig -p | grep -q "$lib"; then
        echo -e "${GREEN}AVAILABLE${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}NOT AVAILABLE${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

echo -e "${YELLOW}System Information:${NC}"
echo "OS: $(uname -s)"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"
echo ""

echo -e "${YELLOW}GPU Driver Status:${NC}"

# Check GPU drivers
if command -v nvidia-smi >/dev/null 2>&1; then
    echo -e "${GREEN}NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | head -5
elif command -v rocminfo >/dev/null 2>&1; then
    echo -e "${GREEN}AMD GPU detected${NC}"
    rocminfo | head -10
elif command -v clinfo >/dev/null 2>&1; then
    echo -e "${GREEN}OpenCL platform detected${NC}"
    clinfo | grep "Platform Name" -A 5 | head -10
else
    echo -e "${RED}No GPU driver detected${NC}"
    ((TESTS_FAILED++))
fi

echo ""
echo -e "${YELLOW}Development Tools:${NC}"

# Check compilers
check_command "gcc" "GCC compiler"
check_command "g++" "G++ compiler"
check_command "cmake" "CMake"
check_command "make" "Make"
check_command "pkg-config" "pkg-config"

# Check GPU-specific tools
echo ""
echo -e "${YELLOW}GPU Development Tools:${NC}"

check_command "nvcc" "NVIDIA CUDA compiler"
check_command "clinfo" "OpenCL info tool"
check_command "vulkaninfo" "Vulkan info tool"

echo ""
echo -e "${YELLOW}GPU Libraries:${NC}"

# Check CUDA libraries
check_library "libcudart.so" "CUDA Runtime"
check_library "libcufft.so" "cuFFT library"
check_library "libcublas.so" "cuBLAS library"
check_library "libcuda.so" "CUDA driver library"

# Check OpenCL libraries
check_library "libOpenCL.so" "OpenCL library"
check_library "libclfft.so" "clFFT library (optional)"

# Check Vulkan libraries
check_library "libvulkan.so" "Vulkan library"

echo ""
echo -e "${YELLOW}Audio Processing Libraries:${NC}"

# Check audio libraries
check_library "libfftw3f.so" "FFTW3 (single precision)"
check_library "libsndfile.so" "libsndfile"
check_library "libasound.so" "ALSA (Linux audio)"
check_library "libpulse.so" "PulseAudio"

echo ""
echo -e "${YELLOW}Configuration Files:${NC}"

# Check configuration files
check_file "config/gpu/cuda_config.json" "GPU configuration"
check_file "config/default.json.in" "Default configuration template"

echo ""
echo -e "${YELLOW}Environment Variables:${NC}"

# Check environment variables
check_env_var() {
    local var="$1"
    local description="$2"

    echo -n "Checking $description... "

    if [ -n "${!var}" ]; then
        echo -e "${GREEN}SET (${!var})${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${YELLOW}NOT SET${NC}"
        ((TESTS_PASSED++)) # Not critical if not set
    fi
}

check_env_var "CUDA_HOME" "CUDA_HOME"
check_env_var "CUDA_PATH" "CUDA_PATH"
check_env_var "LD_LIBRARY_PATH" "LD_LIBRARY_PATH"
check_env_var "VORTEX_GPU_BACKEND" "VORTEX_GPU_BACKEND"

echo ""
echo -e "${YELLOW}Functionality Tests:${NC}"

# Test CUDA compilation if available
if command -v nvcc >/dev/null 2>&1; then
    echo "Testing CUDA compilation..."
    cat > /tmp/test_cuda.cu << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA devices\n", deviceCount);
    return deviceCount > 0 ? 0 : 1;
}
EOF

    if nvcc /tmp/test_cuda.cu -o /tmp/test_cuda 2>/dev/null; then
        if /tmp/test_cuda >/dev/null 2>&1; then
            echo -e "CUDA compilation and execution... ${GREEN}PASS${NC}"
            ((TESTS_PASSED++))
        else
            echo -e "CUDA execution... ${RED}FAIL${NC}"
            ((TESTS_FAILED++))
        fi
        rm -f /tmp/test_cuda
    else
        echo -e "CUDA compilation... ${RED}FAIL${NC}"
        ((TESTS_FAILED++))
    fi
    rm -f /tmp/test_cuda.cu
else
    echo "Skipping CUDA test (nvcc not available)"
fi

# Test OpenCL if available
if command -v clinfo >/dev/null 2>&1; then
    if clinfo >/dev/null 2>&1; then
        echo -e "OpenCL runtime... ${GREEN}PASS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "OpenCL runtime... ${RED}FAIL${NC}"
        ((TESTS_FAILED++))
    fi
else
    echo "Skipping OpenCL test (clinfo not available)"
fi

# Test build system
echo ""
echo -e "${YELLOW}Build System Test:${NC}"

if [ -f "CMakeLists.txt" ]; then
    echo "Testing CMake configuration..."
    if mkdir -p build_test && cd build_test; then
        if cmake .. -DENABLE_GPU=ON >/dev/null 2>&1; then
            echo -e "CMake configuration... ${GREEN}PASS${NC}"
            ((TESTS_PASSED++))
        else
            echo -e "CMake configuration... ${RED}FAIL${NC}"
            ((TESTS_FAILED++))
        fi
        cd ..
        rm -rf build_test
    else
        echo -e "Build directory creation... ${RED}FAIL${NC}"
        ((TESTS_FAILED++))
    fi
else
    echo "CMakeLists.txt not found"
    ((TESTS_FAILED++))
fi

echo ""
echo -e "${YELLOW}Performance Benchmarks:${NC}"

# Run simple GPU performance test if available
if command -v nvcc >/dev/null 2>&1; then
    echo "Running GPU performance test..."
    cat > /tmp/gpu_perf_test.cu << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1024 * 1024 * 32; // 32M elements
    size_t bytes = n * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 0.001f;
        h_b[i] = i * 0.002f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    auto start = std::chrono::high_resolution_clock::now();
    vector_add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double bandwidth = (3.0 * bytes) / (duration.count() * 1e-6) / (1024.0 * 1024.0 * 1024.0);

    printf("GPU bandwidth: %.2f GB/s\n", bandwidth);

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return bandwidth > 10.0 ? 0 : 1;  // Expect >10 GB/s
}
EOF

    if nvcc -O3 -arch=sm_60 /tmp/gpu_perf_test.cu -o /tmp/gpu_perf_test 2>/dev/null; then
        if /tmp/gpu_perf_test 2>/dev/null; then
            echo -e "GPU performance test... ${GREEN}PASS${NC}"
            ((TESTS_PASSED++))
        else
            echo -e "GPU performance test... ${YELLOW}LOW PERFORMANCE${NC}"
            ((TESTS_PASSED++)) # Low performance is still a pass
        fi
        rm -f /tmp/gpu_perf_test
    else
        echo -e "GPU performance test compilation... ${RED}FAIL${NC}"
        ((TESTS_FAILED++))
    fi
    rm -f /tmp/gpu_perf_test.cu
fi

echo ""
echo -e "${BLUE}Summary:${NC}"
echo "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
echo "Tests failed: ${RED}$TESTS_FAILED${NC}"
echo "Total tests: $((TESTS_PASSED + TESTS_FAILED))"

if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ All GPU setup verification tests passed!${NC}"
    echo "Your GPU development environment is properly configured."
    exit 0
else
    echo ""
    echo -e "${RED}❌ $TESTS_FAILED test(s) failed${NC}"
    echo "Please address the issues above before proceeding with GPU development."
    exit 1
fi