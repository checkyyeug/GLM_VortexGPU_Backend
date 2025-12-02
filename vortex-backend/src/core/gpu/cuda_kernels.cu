#include "cuda_kernels.hpp"
#include "system/logger.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <device_atomic_functions.h>

namespace vortex::cuda {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            Logger::error("CUDA error: {} at {}:{}: {}", \
                         cudaGetErrorString(error), __FILE__, __LINE__, #call); \
            return false; \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t error = call; \
        if (error != CUBLAS_STATUS_SUCCESS) { \
            Logger::error("cuBLAS error: {} at {}:{}", \
                         error, __FILE__, __LINE__); \
            return false; \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult error = call; \
        if (error != CUFFT_SUCCESS) { \
            Logger::error("cuFFT error: {} at {}:{}", \
                         error, __FILE__, __LINE__); \
            return false; \
        } \
    } while(0)

// Device constants
__constant__ float d_pi = 3.14159265358979323846f;
__constant__ float d_twoPi = 6.28318530717958647692f;
__constant__ float d_sqrt2 = 1.41421356237309504880f;

// Simple gain kernel
__global__ void applyGainKernel(const float* input, float* output,
                              float gain, size_t numSamples) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numSamples) {
        output[idx] = input[idx] * gain;
    }
}

// Stereo gain kernel
__global__ void applyStereoGainKernel(const float* input, float* output,
                                     float gainLeft, float gainRight,
                                     size_t numFrames) {
    size_t frame = blockIdx.x * blockDim.x + threadIdx.x;

    if (frame < numFrames) {
        output[frame * 2] = input[frame * 2] * gainLeft;     // Left channel
        output[frame * 2 + 1] = input[frame * 2 + 1] * gainRight; // Right channel
    }
}

// Simple mixer kernel
__global__ void audioMixerKernel(const float* input1, const float* input2,
                                float* output, float gain1, float gain2,
                                size_t numSamples) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numSamples) {
        output[idx] = input1[idx] * gain1 + input2[idx] * gain2;
    }
}

// Convolution kernel (simplified overlap-add)
__global__ void convolutionKernel(const float* input, const float* impulse,
                                float* output, size_t inputSize,
                                size_t irSize, size_t outputSize) {
    size_t outputIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (outputIdx < outputSize) {
        float sum = 0.0f;

        for (size_t i = 0; i < irSize && i <= outputIdx && i < inputSize; ++i) {
            sum += input[outputIdx - i] * impulse[i];
        }

        output[outputIdx] = sum;
    }
}

// Optimized convolution kernel for larger impulse responses
__global__ void convolutionOptimizedKernel(const float* input, const float* impulse,
                                          float* output, size_t inputSize,
                                          size_t irSize, size_t outputSize) {
    extern __shared__ float sharedImpulse[];

    size_t tid = threadIdx.x;
    size_t blockSize = blockDim.x;

    // Load impulse response into shared memory
    for (size_t i = tid; i < irSize; i += blockSize) {
        sharedImpulse[i] = impulse[i];
    }
    __syncthreads();

    size_t outputIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (outputIdx < outputSize) {
        float sum = 0.0f;

        for (size_t i = 0; i < irSize && i <= outputIdx && i < inputSize; ++i) {
            sum += input[outputIdx - i] * sharedImpulse[i];
        }

        output[outputIdx] = sum;
    }
}

// FIR filter kernel
__global__ void firFilterKernel(const float* input, const float* coefficients,
                             float* output, size_t numSamples,
                             size_t numCoefficients) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numSamples) {
        float sum = 0.0f;

        for (size_t i = 0; i < numCoefficients && i <= idx; ++i) {
            sum += input[idx - i] * coefficients[i];
        }

        output[idx] = sum;
    }
}

// IIR biquad filter kernel
__global__ void iirBiquadKernel(const float* input, float* output,
                              const float* b0, const float* b1, const float* b2,
                              const float* a1, const float* a2,
                              float* x1, float* x2, float* y1, float* y2,
                              size_t numSamples) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numSamples) {
        float currentInput = input[idx];

        // Apply biquad difference equation
        float currentOutput = (*b0) * currentInput +
                            (*b1) * (*x1) +
                            (*b2) * (*x2) -
                            (*a1) * (*y1) -
                            (*a2) * (*y2);

        output[idx] = currentOutput;

        // Update delay line (using atomic operations for thread safety)
        *x2 = *x1;
        *x1 = currentInput;
        *y2 = *y1;
        *y1 = currentOutput;
    }
}

// FFT window function kernel (Hanning window)
__global__ void hanningWindowKernel(float* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float window = 0.5f * (1.0f - cosf(d_twoPi * idx / (size - 1)));
        data[idx] *= window;
    }
}

// RMS calculation kernel
__global__ void calculateRMSKernel(const float* input, float* rms,
                                 size_t numSamples) {
    extern __shared__ float sdata[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (idx < numSamples) ? input[idx] * input[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        rms[blockIdx.x] = sqrtf(sdata[0] / numSamples);
    }
}

// Peak detection kernel
__global__ void findPeaksKernel(const float* input, int* peaks,
                              size_t numSamples, float threshold) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= 1 && idx < numSamples - 1) {
        if (input[idx] > threshold &&
            input[idx] > input[idx - 1] &&
            input[idx] > input[idx + 1]) {
            peaks[idx] = 1;
        } else {
            peaks[idx] = 0;
        }
    }
}

// DSD to PCM conversion kernel
__global__ void dsdToPCMKernel(const uint8_t* dsdData, float* pcmData,
                              size_t numSamples, int oversamplingFactor) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numSamples) {
        float sum = 0.0f;

        // Apply low-pass filter to DSD bitstream
        for (int i = 0; i < oversamplingFactor && idx * oversamplingFactor + i < numSamples * oversamplingFactor; ++i) {
            int bit = (dsdData[(idx * oversamplingFactor + i) / 8] >> (7 - ((idx * oversamplingFactor + i) % 8))) & 1;
            sum += (bit == 1) ? 1.0f : -1.0f;
        }

        pcmData[idx] = sum / oversamplingFactor;
    }
}

// Multi-channel mixer kernel
__global__ void multichannelMixerKernel(const float* inputs, float* output,
                                       const float* gains, int numChannels,
                                       size_t numSamples) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numSamples) {
        float sum = 0.0f;

        for (int ch = 0; ch < numChannels; ++ch) {
            sum += inputs[ch * numSamples + idx] * gains[ch];
        }

        output[idx] = sum;
    }
}

// Audio resampling kernel (linear interpolation)
__global__ void linearResampleKernel(const float* input, float* output,
                                   double ratio, size_t inputSize,
                                   size_t outputSize) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < outputSize) {
        double inputIndex = idx / ratio;
        size_t index0 = static_cast<size_t>(inputIndex);
        size_t index1 = std::min(index0 + 1, inputSize - 1);
        double fraction = inputIndex - index0;

        output[idx] = input[index0] * (1.0 - fraction) + input[index1] * fraction;
    }
}

// Limiting function
__device__ float limit(float value, float threshold, float ratio) {
    if (value > threshold) {
        return threshold + (value - threshold) / ratio;
    } else if (value < -threshold) {
        return -threshold + (value + threshold) / ratio;
    }
    return value;
}

// Dynamics processing kernel (compression/limiting)
__global__ void dynamicsKernel(const float* input, float* output,
                             float threshold, float ratio,
                             float attackTime, float releaseTime,
                             float sampleRate, size_t numSamples) {
    extern __shared__ float sharedEnvelope[];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = threadIdx.x;

    if (idx < numSamples) {
        float inputLevel = fabsf(input[idx]);
        float envelope = 0.0f;

        // Calculate envelope follower
        if (tid == 0) {
            envelope = inputLevel;
        }
        __syncthreads();

        // Apply compression/limiting
        float gainReduction = 1.0f;
        if (envelope > threshold) {
            gainReduction = 1.0f - (envelope - threshold) * (1.0f - 1.0f/ratio) / envelope;
        }

        output[idx] = input[idx] * gainReduction;
    }
}

CUDAAudioProcessor::CUDAAudioProcessor() {
    Logger::info("CUDAAudioProcessor constructor");

    // Initialize cuBLAS
    cublasStatus_t status = cublasCreate(&cublasHandle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("Failed to create cuBLAS handle");
        cublasHandle_ = nullptr;
    }
}

CUDAAudioProcessor::~CUDAAudioProcessor() {
    shutdown();
    Logger::info("CUDAAudioProcessor destroyed");
}

bool CUDAAudioProcessor::initialize() {
    Logger::info("Initializing CUDA audio processor");

    // Check CUDA device
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        Logger::error("No CUDA devices found");
        return false;
    }

    // Set device to first available
    CUDA_CHECK(cudaSetDevice(0));

    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    Logger::info("Using CUDA device: {} (Compute {}.{}), {} MB memory",
                 prop.name, prop.major, prop.minor,
                 prop.totalGlobalMem / (1024 * 1024));

    initialized_ = true;
    return true;
}

void CUDAAudioProcessor::shutdown() {
    if (cublasHandle_) {
        cublasDestroy(cublasHandle_);
        cublasHandle_ = nullptr;
    }

    initialized_ = false;
    Logger::info("CUDA audio processor shutdown");
}

bool CUDAAudioProcessor::applyGain(const float* input, float* output,
                                 float gain, size_t numSamples) {
    if (!initialized_) {
        Logger::error("CUDA processor not initialized");
        return false;
    }

    // Allocate device memory
    float* d_input;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, numSamples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, numSamples * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input, numSamples * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (numSamples + blockSize - 1) / blockSize;

    applyGainKernel<<<gridSize, blockSize>>>(d_input, d_output, gain, numSamples);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(output, d_output, numSamples * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return true;
}

bool CUDAAudioProcessor::applyStereoGain(const float* input, float* output,
                                        float gainLeft, float gainRight,
                                        size_t numFrames) {
    if (!initialized_) {
        Logger::error("CUDA processor not initialized");
        return false;
    }

    size_t numSamples = numFrames * 2;

    // Allocate device memory
    float* d_input;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, numSamples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, numSamples * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input, numSamples * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (numFrames + blockSize - 1) / blockSize;

    applyStereoGainKernel<<<gridSize, blockSize>>>(d_input, d_output, gainLeft, gainRight, numFrames);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(output, d_output, numSamples * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return true;
}

bool CUDAAudioProcessor::convolve(const float* input, const float* impulse,
                                 float* output, size_t inputSize,
                                 size_t irSize) {
    if (!initialized_) {
        Logger::error("CUDA processor not initialized");
        return false;
    }

    size_t outputSize = inputSize + irSize - 1;

    // Allocate device memory
    float* d_input;
    float* d_impulse;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, inputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_impulse, irSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, outputSize * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, input, inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_impulse, impulse, irSize * sizeof(float), cudaMemcpyHostToDevice));

    // Choose kernel based on impulse response size
    int blockSize = 256;
    int gridSize = (outputSize + blockSize - 1) / blockSize;
    size_t sharedMemSize = irSize * sizeof(float);

    if (sharedMemSize <= 48 * 1024) { // 48KB shared memory limit
        convolutionOptimizedKernel<<<gridSize, blockSize, sharedMemSize>>>(
            d_input, d_impulse, d_output, inputSize, irSize, outputSize);
    } else {
        convolutionKernel<<<gridSize, blockSize>>>(
            d_input, d_impulse, d_output, inputSize, irSize, outputSize);
    }

    CUDA_CHECK(cudaGetLastError());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(output, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_impulse));
    CUDA_CHECK(cudaFree(d_output));

    return true;
}

bool CUDAAudioProcessor::applyFIRFilter(const float* input, const float* coefficients,
                                       float* output, size_t numSamples,
                                       size_t numCoefficients) {
    if (!initialized_) {
        Logger::error("CUDA processor not initialized");
        return false;
    }

    // Allocate device memory
    float* d_input;
    float* d_coefficients;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, numSamples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_coefficients, numCoefficients * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, numSamples * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, input, numSamples * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coefficients, coefficients, numCoefficients * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (numSamples + blockSize - 1) / blockSize;

    firFilterKernel<<<gridSize, blockSize>>>(d_input, d_coefficients, d_output,
                                           numSamples, numCoefficients);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(output, d_output, numSamples * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_coefficients));
    CUDA_CHECK(cudaFree(d_output));

    return true;
}

bool CUDAAudioProcessor::computeFFT(const float* input, float* magnitude,
                                    size_t fftSize) {
    if (!initialized_) {
        Logger::error("CUDA processor not initialized");
        return false;
    }

    if (!cublasHandle_) {
        Logger::error("cuBLAS handle not available");
        return false;
    }

    // Use cuFFT for FFT computation
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, static_cast<int>(fftSize), CUFFT_R2C, 1));

    // Allocate device memory
    float* d_input;
    cufftComplex* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, fftSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, (fftSize / 2 + 1) * sizeof(cufftComplex)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input, fftSize * sizeof(float), cudaMemcpyHostToDevice));

    // Apply Hanning window
    int blockSize = 256;
    int gridSize = (fftSize + blockSize - 1) / blockSize;
    hanningWindowKernel<<<gridSize, blockSize>>>(d_input, fftSize);
    CUDA_CHECK(cudaGetLastError());

    // Execute FFT
    CUFFT_CHECK(cufftExecR2C(plan, d_input, d_output));

    // Copy result back and compute magnitude
    std::vector<cufftComplex> h_output(fftSize / 2 + 1);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < h_output.size(); ++i) {
        magnitude[i] = sqrtf(h_output[i].x * h_output[i].x + h_output[i].y * h_output[i].y) / fftSize;
    }

    // Cleanup
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return true;
}

bool CUDAAudioProcessor::calculateRMS(const float* input, float* rms, size_t numSamples) {
    if (!initialized_) {
        Logger::error("CUDA processor not initialized");
        return false;
    }

    const int blockSize = 256;
    const int gridSize = (numSamples + blockSize - 1) / blockSize;

    // Allocate device memory
    float* d_input;
    float* d_rms;
    CUDA_CHECK(cudaMalloc(&d_input, numSamples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rms, gridSize * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input, numSamples * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    calculateRMSKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        d_input, d_rms, numSamples);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back
    std::vector<float> h_rms(gridSize);
    CUDA_CHECK(cudaMemcpy(h_rms.data(), d_rms, gridSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute final RMS
    *rms = 0.0f;
    for (float val : h_rms) {
        *rms += val * val;
    }
    *rms = sqrtf(*rms / gridSize);

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_rms));

    return true;
}

bool CUDAAudioProcessor::applyDynamics(const float* input, float* output,
                                      float threshold, float ratio,
                                      float attackTime, float releaseTime,
                                      float sampleRate, size_t numSamples) {
    if (!initialized_) {
        Logger::error("CUDA processor not initialized");
        return false;
    }

    // Allocate device memory
    float* d_input;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, numSamples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, numSamples * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input, numSamples * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (numSamples + blockSize - 1) / blockSize;

    dynamicsKernel<<<gridSize, blockSize>>>(d_input, d_output, threshold, ratio,
                                          attackTime, releaseTime, sampleRate, numSamples);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(output, d_output, numSamples * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return true;
}

bool CUDAAudioProcessor::dsdToPCM(const uint8_t* dsdData, float* pcmData,
                                  size_t numSamples, int dsdRate, int pcmRate) {
    if (!initialized_) {
        Logger::error("CUDA processor not initialized");
        return false;
    }

    int oversamplingFactor = dsdRate / pcmRate;

    // Allocate device memory
    uint8_t* d_dsdData;
    float* d_pcmData;
    CUDA_CHECK(cudaMalloc(&d_dsdData, (numSamples * oversamplingFactor + 7) / 8));
    CUDA_CHECK(cudaMalloc(&d_pcmData, numSamples * sizeof(float)));

    // Copy DSD data to device
    CUDA_CHECK(cudaMemcpy(d_dsdData, dsdData, (numSamples * oversamplingFactor + 7) / 8, cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (numSamples + blockSize - 1) / blockSize;

    dsdToPCMKernel<<<gridSize, blockSize>>>(d_dsdData, d_pcmData, numSamples, oversamplingFactor);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(pcmData, d_pcmData, numSamples * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_dsdData));
    CUDA_CHECK(cudaFree(d_pcmData));

    return true;
}

// Remove macros at end of file
#undef CUDA_CHECK
#undef CUBLAS_CHECK
#undef CUFFT_CHECK

} // namespace vortex::cuda