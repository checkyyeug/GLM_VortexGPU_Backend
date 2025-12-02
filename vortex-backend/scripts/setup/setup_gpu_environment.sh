#!/bin/bash

# GPU Development Environment Setup Script
# This script configures the GPU development environment for Vortex Audio Backend

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Vortex GPU Audio Backend - GPU Development Environment Setup${NC}"
echo "=================================================="

# Function to check command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check GPU driver
check_gpu_driver() {
    echo -e "${YELLOW}Checking GPU driver...${NC}"

    if command_exists nvidia-smi; then
        echo -e "${GREEN}NVIDIA driver found:${NC}"
        nvidia-smi
        return 0
    elif command_exists rocminfo; then
        echo -e "${GREEN}AMD driver found:${NC}"
        rocminfo
        return 0
    elif command_exists clinfo; then
        echo -e "${GREEN}OpenCL platform found:${NC}"
        clinfo | head -20
        return 0
    else
        echo -e "${RED}No GPU driver detected${NC}"
        return 1
    fi
}

# Function to install CUDA
install_cuda() {
    echo -e "${YELLOW}Installing CUDA...${NC}"

    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command_exists apt-get; then
            echo "Detected Ubuntu/Debian-based system"

            # Add CUDA repository
            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
            sudo dpkg -i cuda-keyring_1.0-1_all.deb
            sudo apt-get update

            # Install CUDA toolkit
            sudo apt-get -y install cuda-toolkit-12-0 cuda-cudart-dev-12-0

            # Install development libraries
            sudo apt-get -y install cuda-cccl cuda-cufft-dev cuda-cublas-dev

            # Add to PATH
            echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
            echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

        elif command_exists yum; then
            echo "Detected RHEL/CentOS system"
            # RHEL/CentOS installation
            sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
            sudo yum clean all
            sudo yum -y install cuda-toolkit-12-0
        fi

    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "${RED}CUDA is not supported on macOS${NC}"
        echo "Consider using OpenCL or Metal for GPU acceleration on macOS"
        return 1

    else
        echo -e "${RED}Unsupported operating system${NC}"
        return 1
    fi

    echo -e "${GREEN}CUDA installation completed${NC}"
}

# Function to install OpenCL
install_opencl() {
    echo -e "${YELLOW}Installing OpenCL...${NC}"

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command_exists apt-get; then
            sudo apt-get -y install opencl-headers ocl-icd-opencl-dev

            # Install vendor-specific OpenCL implementations
            if command_exists nvidia-smi; then
                sudo apt-get -y install nvidia-opencl-dev
            elif command_exists rocminfo; then
                sudo apt-get -y install rocm-opencl-dev
            else
                sudo apt-get -y install intel-opencl-icd
            fi

        elif command_exists yum; then
            sudo yum -y install opencl-headers ocl-icd-devel
        fi

    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "OpenCL is pre-installed on macOS"

    fi

    echo -e "${GREEN}OpenCL installation completed${NC}"
}

# Function to install Vulkan
install_vulkan() {
    echo -e "${YELLOW}Installing Vulkan...${NC}"

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command_exists apt-get; then
            sudo apt-get -y install vulkan-tools vulkan-sdk vulkan-validationlayers-dev

        elif command_exists yum; then
            sudo yum -y install vulkan-loader-devel vulkan-validationlayers-devel
        fi

    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Vulkan is not supported on macOS"
        echo "Consider using Metal for GPU acceleration on macOS"
        return 1
    fi

    echo -e "${GREEN}Vulkan installation completed${NC}"
}

# Function to verify GPU capabilities
verify_gpu_capabilities() {
    echo -e "${YELLOW}Verifying GPU capabilities...${NC}"

    # Check CUDA
    if command_exists nvcc; then
        echo -e "${GREEN}CUDA compiler found:${NC}"
        nvcc --version

        # Test CUDA compilation
        echo "Testing CUDA compilation..."
        echo '
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s (Compute %d.%d)\n",
               i, prop.name, prop.major, prop.minor);
    }
    return 0;
}' > /tmp/test_cuda.cu

        nvcc /tmp/test_cuda.cu -o /tmp/test_cuda
        /tmp/test_cuda
        rm /tmp/test_cuda.cu /tmp/test_cuda
    fi

    # Check OpenCL
    if command_exists clinfo; then
        echo -e "${GREEN}OpenCL platforms:${NC}"
        clinfo | grep "Platform Name" -A 5
    fi

    # Check Vulkan
    if command_exists vulkaninfo; then
        echo -e "${GREEN}Vulkan devices:${NC}"
        vulkaninfo --summary
    fi
}

# Function to setup GPU environment variables
setup_environment() {
    echo -e "${YELLOW}Setting up environment variables...${NC}"

    # Create GPU environment config
    cat > ~/.vortex_gpu_env << 'EOF'
# Vortex GPU Audio Backend Environment Variables

# CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# OpenCL paths
export OPENCL_LIB_PATH=/usr/lib/x86_64-linux-gnu
export OPENCL_INCLUDE_PATH=/usr/include

# Vulkan paths
export VULKAN_SDK=/usr/local/vulkansdk
export PATH=$VULKAN_SDK/bin:$PATH
export LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH

# Vortex-specific
export VORTEX_GPU_BACKEND=cuda
export VORTEX_GPU_MEMORY_POOL_SIZE=2048
export VORTEX_GPU_ENABLE_PROFILING=false
export VORTEX_GPU_DEBUG=false
EOF

    # Source environment file
    source ~/.vortex_gpu_env

    # Add to bashrc
    echo "" >> ~/.bashrc
    echo "# Vortex GPU Audio Backend Environment" >> ~/.bashrc
    echo "source ~/.vortex_gpu_env" >> ~/.bashrc

    echo -e "${GREEN}Environment variables configured${NC}"
}

# Function to create GPU test
create_gpu_test() {
    echo -e "${YELLOW}Creating GPU test application...${NC}"

    mkdir -p gpu_test
    cd gpu_test

    # Create simple CUDA test
    cat > test_cuda.cu << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void test_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;
    }
}

int main() {
    int size = 1024 * 1024;
    size_t bytes = size * sizeof(float);

    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);

    // Initialize input
    for (int i = 0; i < size; i++) {
        h_input[i] = i * 0.001f;
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Copy to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    test_kernel<<<numBlocks, blockSize>>>(d_input, d_output, size);

    // Copy back to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Verify results
    bool success = true;
    for (int i = 0; i < size; i++) {
        float expected = h_input[i] * 2.0f;
        if (abs(h_output[i] - expected) > 1e-6f) {
            success = false;
            break;
        }
    }

    printf("GPU test %s\n", success ? "PASSED" : "FAILED");

    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return success ? 0 : 1;
}
EOF

    # Create Makefile
    cat > Makefile << 'EOF'
CUDA_ROOT=/usr/local/cuda
NVCC=$(CUDA_ROOT)/bin/nvcc
NVCC_FLAGS=-arch=sm_60 -O3

all: test_cuda

test_cuda: test_cuda.cu
	$(NVCC) $(NVCC_FLAGS) -o test_cuda test_cuda.cu

run: test_cuda
	./test_cuda

clean:
	rm -f test_cuda
EOF

    echo -e "${GREEN}GPU test created in gpu_test/ directory${NC}"
    echo "Run 'cd gpu_test && make run' to test GPU functionality"
    cd ..
}

# Function to check specific requirements for audio processing
check_audio_requirements() {
    echo -e "${YELLOW}Checking audio processing requirements...${NC}"

    # Check if GPU supports required features
    if command_exists nvidia-smi; then
        # Get GPU compute capability
        local compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1)
        echo "GPU Compute Capability: $compute_cap"

        # Check minimum compute capability
        local major=$(echo $compute_cap | cut -d. -f1)
        local minor=$(echo $compute_cap | cut -d. -f2)

        if [[ $major -lt 6 ]]; then
            echo -e "${RED}GPU compute capability $compute_cap is below minimum required (6.0)${NC}"
            return 1
        fi

        echo -e "${GREEN}GPU compute capability $compute_cap meets requirements${NC}"

        # Check memory
        local memory_gb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        memory_gb=$((memory_gb / 1024))
        echo "GPU Memory: ${memory_gb}GB"

        if [[ $memory_gb -lt 4 ]]; then
            echo -e "${YELLOW}GPU memory may be insufficient for large audio processing tasks${NC}"
        fi

        echo -e "${GREEN}GPU memory: ${memory_gb}GB${NC}"
    fi

    # Check for required libraries
    local required_libs=("libfftw3" "libsndfile" "libasound" "libpulse")
    for lib in "${required_libs[@]}"; do
        if ldconfig -p | grep -q $lib; then
            echo -e "${GREEN}✓ $lib found${NC}"
        else
            echo -e "${RED}✗ $lib not found${NC}"
        fi
    done
}

# Main installation flow
main() {
    echo "Starting GPU development environment setup..."

    # Check GPU driver
    if ! check_gpu_driver; then
        echo -e "${RED}No suitable GPU driver found. Please install GPU drivers first.${NC}"
        exit 1
    fi

    # Install GPU APIs
    echo
    read -p "Do you want to install CUDA? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_cuda
    fi

    read -p "Do you want to install OpenCL? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_opencl
    fi

    read -p "Do you want to install Vulkan? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_vulkan
    fi

    # Setup environment
    echo
    setup_environment

    # Verify capabilities
    echo
    verify_gpu_capabilities

    # Check audio requirements
    echo
    check_audio_requirements

    # Create GPU test
    echo
    create_gpu_test

    echo
    echo -e "${GREEN}GPU development environment setup completed!${NC}"
    echo
    echo "Next steps:"
    echo "1. Source the environment: source ~/.vortex_gpu_env"
    echo "2. Test GPU functionality: cd gpu_test && make run"
    echo "3. Build Vortex backend with GPU support: mkdir build && cd build && cmake .. -DENABLE_GPU=ON"
    echo "4. Run the application: ./vortex-backend --gpu-backend cuda"
}

# Check if running with sufficient privileges
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}This script should not be run as root${NC}"
   echo "Run as regular user - it will prompt for sudo when needed"
   exit 1
fi

# Run main function
main "$@"