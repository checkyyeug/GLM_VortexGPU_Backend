#pragma once

#include <cstddef>
#include <cstdint>

namespace vortex::cuda {

/**
 * @brief CUDA audio processing kernels implementation
 *
 * This class provides GPU-accelerated audio processing functions using CUDA.
 * It includes kernels for gain control, convolution, filtering, FFT, dynamics
 * processing, and DSD to PCM conversion. All functions are optimized for
 * real-time audio processing with minimal latency.
 */
class CUDAAudioProcessor {
public:
    CUDAAudioProcessor();
    ~CUDAAudioProcessor();

    // Initialization
    bool initialize();
    void shutdown();

    // Basic audio processing
    bool applyGain(const float* input, float* output, float gain, size_t numSamples);
    bool applyStereoGain(const float* input, float* output,
                        float gainLeft, float gainRight, size_t numFrames);

    // Convolution and filtering
    bool convolve(const float* input, const float* impulse, float* output,
                 size_t inputSize, size_t irSize);
    bool applyFIRFilter(const float* input, const float* coefficients,
                       float* output, size_t numSamples, size_t numCoefficients);

    // Spectral processing
    bool computeFFT(const float* input, float* magnitude, size_t fftSize);

    // Dynamics processing
    bool calculateRMS(const float* input, float* rms, size_t numSamples);
    bool applyDynamics(const float* input, float* output,
                      float threshold, float ratio,
                      float attackTime, float releaseTime,
                      float sampleRate, size_t numSamples);

    // DSD processing
    bool dsdToPCM(const uint8_t* dsdData, float* pcmData,
                  size_t numSamples, int dsdRate, int pcmRate);

    // Utility functions
    bool copyToGPU(const void* hostPtr, void* devicePtr, size_t size);
    bool copyFromGPU(const void* devicePtr, void* hostPtr, size_t size);
    void* allocateGPUMemory(size_t size);
    void deallocateGPUMemory(void* ptr);

private:
    bool initialized_ = false;
    void* cublasHandle_ = nullptr;

    // Internal helper methods
    bool checkCUDAError(cudaError_t error, const char* operation);
    bool checkCUBLASError(cublasStatus_t error, const char* operation);
    bool checkCUFFTError(cufftResult error, const char* operation);
};

} // namespace vortex::cuda