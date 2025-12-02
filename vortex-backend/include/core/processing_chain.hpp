#pragma once

#include "audio_processor.hpp"
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <atomic>

namespace vortex {

struct ParameterState {
    std::string name;
    float value;
    float defaultValue;
    float min;
    float max;
};

struct ProcessorState {
    size_t id;
    std::string name;
    bool enabled;
    bool bypassed;
    std::vector<ParameterState> parameters;
};

struct ChainState {
    double sampleRate;
    int bufferSize;
    int channels;
    bool gpuEnabled;
    size_t numProcessors;
    std::vector<ProcessorState> processors;
};

/**
 * @brief Real-time audio processing pipeline
 *
 * This class manages a chain of audio processors that process audio data
 * in real-time. It supports both CPU and GPU processing, parameter management,
 * and state serialization. The processing pipeline is thread-safe and optimized
 * for low-latency audio applications.
 */
class ProcessingChain {
public:
    ProcessingChain();
    ~ProcessingChain();

    // Initialization
    bool initialize(double sampleRate = 44100.0, int bufferSize = 512, int channels = 2);
    void shutdown();

    // Processor management
    size_t addProcessor(std::unique_ptr<AudioProcessor> processor);
    bool removeProcessor(size_t processorId);
    AudioProcessor* getProcessor(size_t processorId);
    std::vector<size_t> getProcessorIds() const;

    // Processing methods
    void processCPU(float* audioData, size_t numSamples);
    void processGPU(float* audioData, size_t numSamples, class GPUProcessor* gpuProcessor);

    // GPU acceleration
    bool enableGPUAcceleration(class GPUProcessor* gpuProcessor);
    void disableGPUAcceleration();

    // Control
    void reset();
    void setBypass(bool bypass);

    // State management
    ChainState getState() const;
    bool loadState(const ChainState& state);

    // Getters
    bool isInitialized() const;
    bool isGPUEnabled() const;
    double getSampleRate() const;
    int getBufferSize() const;
    int getChannels() const;
    size_t getNumProcessors() const;

private:
    // Core state
    bool initialized_;
    double sampleRate_;
    int bufferSize_;
    int channels_;
    bool gpuEnabled_;

    // Processor management
    mutable std::mutex processorsMutex_;
    std::map<size_t, std::unique_ptr<AudioProcessor>> processors_;
    std::vector<size_t> processorOrder_;
    std::atomic<size_t> processorIdCounter_;

    // GPU processor reference
    class GPUProcessor* gpuProcessor_;

    // Private methods
    void setupDefaultChain();
    void setupDefaultEqualizer(class EqualizerProcessor* equalizer);
    size_t addProcessorInternal(std::unique_ptr<AudioProcessor> processor);
};

} // namespace vortex