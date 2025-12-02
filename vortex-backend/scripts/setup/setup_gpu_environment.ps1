# GPU Development Environment Setup Script for Windows PowerShell
# This script configures the GPU development environment for Vortex Audio Backend on Windows

param(
    [switch]$InstallCUDA,
    [switch]$InstallOpenCL,
    [switch]$SkipDriverCheck,
    [string]$CUDAVersion = "12.0"
)

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )

    Write-Host $Message -ForegroundColor $Color
}

function Test-Command {
    param([string]$Command)

    try {
        $null = Get-Command $Command -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

function Test-GPUDriver {
    Write-ColorOutput "Checking GPU driver..." "Yellow"

    if (Test-Command "nvidia-smi") {
        Write-ColorOutput "NVIDIA driver found:" "Green"
        nvidia-smi
        return $true
    }
    elseif (Test-Command "rocminfo") {
        Write-ColorOutput "AMD driver found:" "Green"
        rocminfo
        return $true
    }
    else {
        Write-ColorOutput "No GPU driver detected" "Red"
        return $false
    }
}

function Install-CUDAToolkit {
    param([string]$Version = "12.0")

    Write-ColorOutput "Installing CUDA Toolkit $Version..." "Yellow"

    $url = "https://developer.download.nvidia.com/compute/cuda/${Version}.0/local_installers/cuda_${Version}.0.528.33_windows.exe"
    $installer = "$env:TEMP\cuda_installer.exe"

    try {
        Write-ColorOutput "Downloading CUDA Toolkit..." "Yellow"
        Invoke-WebRequest -Uri $url -OutFile $installer

        Write-ColorOutput "Running CUDA installer..." "Yellow"
        # Run installer silently
        Start-Process -FilePath $installer -ArgumentList "-s" -Wait

        # Set environment variables
        [Environment]::SetEnvironmentVariable("CUDA_HOME", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$Version", "User")
        [Environment]::SetEnvironmentVariable("CUDA_PATH", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$Version", "User")

        $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$Version\bin"
        $env:PATH += ";$cudaPath"
        [Environment]::SetEnvironmentVariable("PATH", $env:PATH, "User")

        $cudaLibPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$Version\libnvvp"
        $env:PATH += ";$cudaLibPath"
        [Environment]::SetEnvironmentVariable("PATH", $env:PATH, "User")

        Write-ColorOutput "CUDA Toolkit installation completed" "Green"
        return $true
    }
    catch {
        Write-ColorOutput "CUDA installation failed: $_" "Red"
        return $false
    }
    finally {
        if (Test-Path $installer) {
            Remove-Item $installer -Force
        }
    }
}

function Install-OpenCLSDK {
    Write-ColorOutput "Installing OpenCL SDK..." "Yellow"

    try {
        # Install Visual Studio with C++ workload if not present
        if (-not (Test-Command "cl.exe")) {
            Write-ColorOutput "Visual Studio Build Tools not found. Please install Visual Studio with C++ workload." "Yellow"
            Write-ColorOutput "Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022" "Yellow"
        }

        # Install Intel OpenCL SDK
        $intelSdkUrl = "https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.12572.8/opencl-intel-cpu-2023.1.30.0.0_rel-win.zip"
        $intelSdkZip = "$env:TEMP\intel_opencl_sdk.zip"
        $extractPath = "$env:TEMP\intel_opencl"

        Write-ColorOutput "Downloading Intel OpenCL SDK..." "Yellow"
        Invoke-WebRequest -Uri $intelSdkUrl -OutFile $intelSdkZip

        Expand-Archive -Path $intelSdkZip -DestinationPath $extractPath -Force

        # Copy SDK files to a permanent location
        $sdkPath = "C:\IntelOpenCLSDK"
        if (-not (Test-Path $sdkPath)) {
            New-Item -ItemType Directory -Path $sdkPath -Force
        }

        Copy-Item -Path "$extractPath\*" -Destination $sdkPath -Recurse -Force

        # Set environment variables
        [Environment]::SetEnvironmentVariable("INTEL_OPENCL_SDK", $sdkPath, "User")
        $env:PATH += ";$sdkPath\bin"
        [Environment]::SetEnvironmentVariable("PATH", $env:PATH, "User")

        Write-ColorOutput "Intel OpenCL SDK installation completed" "Green"
        return $true
    }
    catch {
        Write-ColorOutput "OpenCL SDK installation failed: $_" "Red"
        return $false
    }
    finally {
        if (Test-Path $intelSdkZip) {
            Remove-Item $intelSdkZip -Force
        }
        if (Test-Path $extractPath) {
            Remove-Item $extractPath -Recurse -Force
        }
    }
}

function Test-GPUCapabilities {
    Write-ColorOutput "Verifying GPU capabilities..." "Yellow"

    # Test CUDA
    if (Test-Command "nvcc") {
        Write-ColorOutput "CUDA compiler found:" "Green"
        nvcc --version

        # Test CUDA compilation
        Write-ColorOutput "Testing CUDA compilation..." "Yellow"

        $testCode = @"
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
}
"@

        Set-Content -Path "$env:TEMP\test_cuda.cu" -Value $testCode

        try {
            $cudaPath = (Get-Command nvcc).Source | Split-Path -Parent
            $nvccPath = Join-Path $cudaPath "..\..\nvcc.exe"

            & $nvccPath "$env:TEMP\test_cuda.cu" -o "$env:TEMP\test_cuda.exe"
            & "$env:TEMP\test_cuda.exe"

            Write-ColorOutput "CUDA compilation test passed" "Green"
        }
        catch {
            Write-ColorOutput "CUDA compilation test failed: $_" "Red"
        }
        finally {
            Remove-Item "$env:TEMP\test_cuda.cu" -ErrorAction SilentlyContinue
            Remove-Item "$env:TEMP\test_cuda.exe" -ErrorAction SilentlyContinue
        }
    }

    # Test OpenCL
    if (Get-ChildItem "HKLM:\SOFTWARE\Khronos\OpenCL" -ErrorAction SilentlyContinue) {
        Write-ColorOutput "OpenCL platforms found:" "Green"
        Get-ChildItem "HKLM:\SOFTWARE\Khronos\OpenCL\Vendors" -ErrorAction SilentlyContinue | ForEach-Object {
            $vendor = $_.Name -replace ".*\\", ""
            Write-Output "  - $vendor"
        }
    }
}

function Setup-Environment {
    Write-ColorOutput "Setting up environment variables..." "Yellow"

    # Create GPU environment config
    $envContent = @"
# Vortex GPU Audio Backend Environment Variables for Windows

# CUDA paths
`$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$CUDAVersion"
`$env:PATH += ";`$env:CUDA_HOME\bin;`$env:CUDA_HOME\libnvvp"

# OpenCL paths
`$env:INTEL_OPENCL_SDK = "C:\IntelOpenCLSDK"
`$env:PATH += ";`$env:INTEL_OPENCL_SDK\bin"

# Vortex-specific
`$env:VORTEX_GPU_BACKEND = "cuda"
`$env:VORTEX_GPU_MEMORY_POOL_SIZE = "2048"
`$env:VORTEX_GPU_ENABLE_PROFILING = "`$false"
`$env:VORTEX_GPU_DEBUG = "`$false"
"@

    $envFile = "$env:USERPROFILE\vortex_gpu_env.ps1"
    Set-Content -Path $envFile -Value $envContent

    # Add to PowerShell profile
    $profilePath = "$env:USERPROFILE\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
    if (-not (Test-Path $profilePath)) {
        New-Item -ItemType File -Path $profilePath -Force
    }

    Add-Content -Path $profilePath -Value "`n# Vortex GPU Audio Backend Environment"
    Add-Content -Path $profilePath -Value ". `"$envFile`""

    Write-ColorOutput "Environment variables configured in $envFile" "Green"
}

function Create-GPUTest {
    Write-ColorOutput "Creating GPU test application..." "Yellow"

    $testDir = "gpu_test"
    if (Test-Path $testDir) {
        Remove-Item $testDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $testDir -Force | Out-Null

    Set-Location $testDir

    # Create CUDA test file
    $cudaTest = @"
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
"@

    Set-Content -Path "test_cuda.cu" -Value $cudaTest

    # Create batch file for compilation
    $batchFile = @"
@echo off
echo Compiling CUDA test...
nvcc -arch=sm_60 -O3 -o test_cuda.exe test_cuda.cu
if errorlevel 1 (
    echo Compilation failed
    exit /b 1
)
echo Running GPU test...
test_cuda.exe
"@

    Set-Content -Path "build_and_run.bat" -Value $batchFile

    Write-ColorOutput "GPU test created in $testDir directory" "Green"
    Write-ColorOutput "Run 'build_and_run.bat' to test GPU functionality" "Yellow"

    Set-Location ..
}

function Test-AudioRequirements {
    Write-ColorOutput "Checking audio processing requirements..." "Yellow"

    # Check GPU capabilities for audio processing
    if (Test-Command "nvidia-smi") {
        $gpuInfo = nvidia-smi --query-gpu=compute_cap,memory.total --format=csv,noheader,nounits | Select-Object -First 1
        $computeCap = $gpuInfo.Split(",")[0].Trim()
        $memoryMB = [int]$gpuInfo.Split(",")[1].Trim()
        $memoryGB = [math]::Round($memoryMB / 1024, 2)

        Write-ColorOutput "GPU Compute Capability: $computeCap" "Green"
        Write-ColorOutput "GPU Memory: ${memoryGB}GB" "Green"

        $major = [int]$computeCap.Split(".")[0]
        if ($major -lt 6) {
            Write-ColorOutput "GPU compute capability $computeCap is below minimum required (6.0)" "Red"
        } else {
            Write-ColorOutput "GPU compute capability $computeCap meets requirements" "Green"
        }

        if ($memoryGB -lt 4) {
            Write-ColorOutput "GPU memory may be insufficient for large audio processing tasks" "Yellow"
        }
    }

    # Check for required audio libraries
    $requiredLibs = @("fftw3", "libsndfile-1", "libportaudio-2")
    foreach ($lib in $requiredLibs) {
        $dllPath = Get-ChildItem -Path "C:\Program Files" -Recurse -Name "$lib.dll" -ErrorAction SilentlyContinue
        if ($dllPath) {
            Write-ColorOutput "✓ $lib found" "Green"
        } else {
            Write-ColorOutput "✗ $lib not found" "Red"
        }
    }
}

# Main script execution
Write-ColorOutput "Vortex GPU Audio Backend - GPU Development Environment Setup for Windows" "Green"
Write-ColorOutput "==============================================================================" "Green"

# Check GPU driver
if (-not $SkipDriverCheck) {
    if (-not (Test-GPUDriver)) {
        Write-ColorOutput "No suitable GPU driver found. Please install GPU drivers first." "Red"
        exit 1
    }
}

# Install GPU APIs
if ($InstallCUDA) {
    Write-Host
    Install-CUDAToolkit -Version $CUDAVersion
}

if ($InstallOpenCL) {
    Write-Host
    Install-OpenCLSDK
}

# Setup environment
Write-Host
Setup-Environment

# Verify capabilities
Write-Host
Test-GPUCapabilities

# Check audio requirements
Write-Host
Test-AudioRequirements

# Create GPU test
Write-Host
Create-GPUTest

Write-Host
Write-ColorOutput "GPU development environment setup completed!" "Green"
Write-Host
Write-ColorOutput "Next steps:" "Yellow"
Write-ColorOutput "1. Restart PowerShell to load environment variables" "Yellow"
Write-ColorOutput "2. Test GPU functionality: cd gpu_test && .\build_and_run.bat" "Yellow"
Write-ColorOutput "3. Build Vortex backend with GPU support:" "Yellow"
Write-ColorOutput "   mkdir build && cd build" "Yellow"
Write-ColorOutput "   cmake .. -DENABLE_GPU=ON -DGPU_BACKEND=CUDA" "Yellow"
Write-ColorOutput "   cmake --build . --config Release" "Yellow"
Write-ColorOutput "4. Run the application: .\Release\vortex-backend.exe --gpu-backend cuda" "Yellow"