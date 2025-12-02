#include "processing_chain.hpp"
#include "system/logger.hpp"

#include <juce_dsp/juce_dsp.h>

namespace vortex {

ProcessingChain::ProcessingChain()
    : initialized_(false)
    , sampleRate_(44100.0)
    , bufferSize_(512)
    , channels_(2)
    , gpuEnabled_(false)
    , processorIdCounter_(0)
{
    Logger::info("ProcessingChain constructor");
}

ProcessingChain::~ProcessingChain() {
    shutdown();
    Logger::info("ProcessingChain destroyed");
}

bool ProcessingChain::initialize(double sampleRate, int bufferSize, int channels) {
    if (initialized_) {
        Logger::warning("ProcessingChain already initialized");
        return true;
    }

    Logger::info("Initializing ProcessingChain: sr={}, bs={}, ch={}",
                 sampleRate, bufferSize, channels);

    try {
        sampleRate_ = sampleRate;
        bufferSize_ = bufferSize;
        channels_ = channels;

        // Initialize DSP context
        dsp::ProcessSpec spec;
        spec.sampleRate = sampleRate;
        spec.maximumBlockSize = bufferSize;
        spec.numChannels = channels;

        // Setup default processing chain
        setupDefaultChain();

        initialized_ = true;
        Logger::info("ProcessingChain initialized successfully with {} processors", processors_.size());

        return true;

    } catch (const std::exception& e) {
        Logger::error("ProcessingChain initialization failed: {}", e.what());
        return false;
    }
}

void ProcessingChain::shutdown() {
    if (!initialized_) {
        return;
    }

    Logger::info("Shutting down ProcessingChain");

    // Clear all processors
    std::lock_guard<std::mutex> lock(processorsMutex_);
    processors_.clear();
    processorOrder_.clear();

    initialized_ = false;
    Logger::info("ProcessingChain shutdown completed");
}

void ProcessingChain::setupDefaultChain() {
    std::lock_guard<std::mutex> lock(processorsMutex_);

    // Add input gain processor
    auto inputGain = std::make_unique<GainProcessor>();
    inputGain->initialize("Input Gain", sampleRate_, bufferSize_, channels_);
    inputGain->setParameter("gain_db", 0.0f);
    addProcessorInternal(std::move(inputGain));

    // Add equalizer processor
    auto equalizer = std::make_unique<EqualizerProcessor>();
    equalizer->initialize("Equalizer", sampleRate_, bufferSize_, channels_);
    setupDefaultEqualizer(equalizer.get());
    addProcessorInternal(std::move(equalizer));

    // Add dynamics processor
    auto dynamics = std::make_unique<DynamicsProcessor>();
    dynamics->initialize("Dynamics", sampleRate_, bufferSize_, channels_);
    dynamics->setParameter("threshold_db", -20.0f);
    dynamics->setParameter("ratio", 4.0f);
    dynamics->setParameter("attack_ms", 5.0f);
    dynamics->setParameter("release_ms", 50.0f);
    addProcessorInternal(std::move(dynamics));

    // Add output gain processor
    auto outputGain = std::make_unique<GainProcessor>();
    outputGain->initialize("Output Gain", sampleRate_, bufferSize_, channels_);
    outputGain->setParameter("gain_db", 0.0f);
    addProcessorInternal(std::move(outputGain));

    Logger::info("Default processing chain setup completed");
}

void ProcessingChain::setupDefaultEqualizer(EqualizerProcessor* equalizer) {
    // Setup 10-band equalizer with standard frequencies
    std::vector<double> frequencies = {
        31.5,   63,   125,  250,  500,
        1000,   2000, 4000, 8000,  16000
    };
    std::vector<double> gains(10, 0.0);  // Flat response
    std::vector<double> q(10, 1.414);     // Standard Q values

    equalizer->configure(frequencies, gains, q);
}

size_t ProcessingChain::addProcessor(std::unique_ptr<AudioProcessor> processor) {
    if (!initialized_) {
        Logger::error("ProcessingChain not initialized");
        return 0;
    }

    if (!processor) {
        Logger::error("Null processor provided");
        return 0;
    }

    std::lock_guard<std::mutex> lock(processorsMutex_);
    return addProcessorInternal(std::move(processor));
}

size_t ProcessingChain::addProcessorInternal(std::unique_ptr<AudioProcessor> processor) {
    size_t id = ++processorIdCounter_;
    processor->setId(id);

    // Store processor
    processors_[id] = std::move(processor);
    processorOrder_.push_back(id);

    Logger::info("Added processor with ID: {}", id);
    return id;
}

bool ProcessingChain::removeProcessor(size_t processorId) {
    std::lock_guard<std::mutex> lock(processorsMutex_);

    auto it = processors_.find(processorId);
    if (it == processors_.end()) {
        Logger::warning("Processor with ID {} not found", processorId);
        return false;
    }

    // Remove from order vector
    processorOrder_.erase(
        std::remove(processorOrder_.begin(), processorOrder_.end(), processorId),
        processorOrder_.end());

    // Remove processor
    processors_.erase(it);

    Logger::info("Removed processor with ID: {}", processorId);
    return true;
}

AudioProcessor* ProcessingChain::getProcessor(size_t processorId) {
    std::lock_guard<std::mutex> lock(processorsMutex_);

    auto it = processors_.find(processorId);
    return (it != processors_.end()) ? it->second.get() : nullptr;
}

std::vector<size_t> ProcessingChain::getProcessorIds() const {
    std::lock_guard<std::mutex> lock(processorsMutex_);
    return processorOrder_;
}

void ProcessingChain::processCPU(float* audioData, size_t numSamples) {
    if (!initialized_) {
        Logger::error("ProcessingChain not initialized");
        return;
    }

    std::lock_guard<std::mutex> lock(processorsMutex_);

    try {
        // Create audio buffer
        juce::AudioBuffer<float> buffer(channels_, static_cast<int>(numSamples));

        // De-interleave input
        for (size_t sample = 0; sample < numSamples; ++sample) {
            for (int channel = 0; channel < channels_; ++channel) {
                buffer.setSample(channel, static_cast<int>(sample),
                               audioData[sample * channels_ + channel]);
            }
        }

        // Process each processor in order
        dsp::AudioBlock<float> block(buffer);
        dsp::ProcessContextReplacing<float> context(block);

        for (size_t processorId : processorOrder_) {
            auto it = processors_.find(processorId);
            if (it != processors_.end()) {
                auto& processor = it->second;
                if (processor->isEnabled()) {
                    processor->process(context);
                }
            }
        }

        // Interleave output
        for (size_t sample = 0; sample < numSamples; ++sample) {
            for (int channel = 0; channel < channels_; ++channel) {
                audioData[sample * channels_ + channel] = buffer.getSample(channel, static_cast<int>(sample));
            }
        }

    } catch (const std::exception& e) {
        Logger::error("CPU processing failed: {}", e.what());
    }
}

void ProcessingChain::processGPU(float* audioData, size_t numSamples, GPUProcessor* gpuProcessor) {
    if (!initialized_ || !gpuProcessor) {
        Logger::error("ProcessingChain or GPU processor not available");
        return;
    }

    std::lock_guard<std::mutex> lock(processorsMutex_);

    try {
        // Prepare GPU processing
        gpuProcessor->beginProcessing(numSamples, channels_);

        // Process each processor, using GPU when available
        for (size_t processorId : processorOrder_) {
            auto it = processors_.find(processorId);
            if (it != processors_.end()) {
                auto& processor = it->second;
                if (processor->isEnabled()) {
                    if (processor->supportsGPU() && gpuProcessor) {
                        processor->processGPU(audioData, numSamples, gpuProcessor);
                    } else {
                        // Fallback to CPU processing
                        processor->processCPU(audioData, numSamples);
                    }
                }
            }
        }

        // Finalize GPU processing
        gpuProcessor->endProcessing();

    } catch (const std::exception& e) {
        Logger::error("GPU processing failed: {}", e.what());
        // Fallback to CPU processing
        processCPU(audioData, numSamples);
    }
}

bool ProcessingChain::enableGPUAcceleration(GPUProcessor* gpuProcessor) {
    if (!gpuProcessor) {
        Logger::error("GPU processor not provided");
        return false;
    }

    std::lock_guard<std::mutex> lock(processorsMutex_);

    gpuEnabled_ = true;
    gpuProcessor_ = gpuProcessor;

    Logger::info("GPU acceleration enabled for processing chain");
    return true;
}

void ProcessingChain::disableGPUAcceleration() {
    std::lock_guard<std::mutex> lock(processorsMutex_);

    gpuEnabled_ = false;
    gpuProcessor_ = nullptr;

    Logger::info("GPU acceleration disabled for processing chain");
}

void ProcessingChain::reset() {
    std::lock_guard<std::mutex> lock(processorsMutex_);

    for (auto& pair : processors_) {
        pair.second->reset();
    }

    Logger::info("All processors reset");
}

void ProcessingChain::setBypass(bool bypass) {
    std::lock_guard<std::mutex> lock(processorsMutex_);

    for (auto& pair : processors_) {
        pair.second->setEnabled(!bypass);
    }

    Logger::info("Processing chain bypass set to: {}", bypass);
}

ProcessingChain::ChainState ProcessingChain::getState() const {
    std::lock_guard<std::mutex> lock(processorsMutex_);

    ChainState state;
    state.sampleRate = sampleRate_;
    state.bufferSize = bufferSize_;
    state.channels = channels_;
    state.gpuEnabled = gpuEnabled_;
    state.numProcessors = processors_.size();

    for (const auto& pair : processors_) {
        ProcessorState processorState;
        processorState.id = pair.first;
        processorState.name = pair.second->getName();
        processorState.enabled = pair.second->isEnabled();
        processorState.bypassed = pair.second->isBypassed();

        // Get all parameters
        for (const auto& param : pair.second->getParameters()) {
            ParameterState paramState;
            paramState.name = param.name;
            paramState.value = param.value;
            paramState.defaultValue = param.defaultValue;
            paramState.min = param.min;
            paramState.max = param.max;
            processorState.parameters.push_back(paramState);
        }

        state.processors.push_back(processorState);
    }

    return state;
}

bool ProcessingChain::loadState(const ChainState& state) {
    if (state.sampleRate != sampleRate_ ||
        state.bufferSize != bufferSize_ ||
        state.channels != channels_) {
        Logger::error("Incompatible state configuration");
        return false;
    }

    std::lock_guard<std::mutex> lock(processorsMutex_);

    // Apply processor states
    for (const auto& processorState : state.processors) {
        auto* processor = getProcessor(processorState.id);
        if (processor) {
            processor->setEnabled(processorState.enabled);
            processor->setBypassed(processorState.bypassed);

            // Apply parameters
            for (const auto& param : processorState.parameters) {
                processor->setParameter(param.name, param.value);
            }
        }
    }

    gpuEnabled_ = state.gpuEnabled;

    Logger::info("Processing chain state loaded successfully");
    return true;
}

// Getters
bool ProcessingChain::isInitialized() const { return initialized_; }
bool ProcessingChain::isGPUEnabled() const { return gpuEnabled_; }
double ProcessingChain::getSampleRate() const { return sampleRate_; }
int ProcessingChain::getBufferSize() const { return bufferSize_; }
int ProcessingChain::getChannels() const { return channels_; }
size_t ProcessingChain::getNumProcessors() const {
    std::lock_guard<std::mutex> lock(processorsMutex_);
    return processors_.size();
}

} // namespace vortex