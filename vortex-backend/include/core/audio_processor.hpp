#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <functional>

namespace vortex {

class GPUProcessor;

struct Parameter {
    std::string name;
    std::string displayName;
    std::string unit;
    float value;
    float defaultValue;
    float min;
    float max;
    float step;
    bool isAutomatable = true;
    std::string category;
};

/**
 * @brief Base class for audio processors in the processing chain
 *
 * This abstract class defines the interface for all audio processors,
 * including effects, dynamics processors, filters, and other DSP components.
 * It supports both CPU and GPU processing with parameter management.
 */
class AudioProcessor {
public:
    AudioProcessor(const std::string& name);
    virtual ~AudioProcessor() = default;

    // Initialization
    virtual bool initialize(const std::string& name, double sampleRate, int bufferSize, int channels) = 0;
    virtual void shutdown() = 0;

    // Processing methods
    virtual void process(juce::dsp::ProcessContextReplacing<float>& context) = 0;
    virtual void processCPU(float* audioData, size_t numSamples) {}
    virtual void processGPU(float* audioData, size_t numSamples, GPUProcessor* gpuProcessor) {}

    // Configuration
    virtual bool supportsGPU() const { return false; }
    virtual void prepare(double sampleRate, int bufferSize, int channels) = 0;
    virtual void reset() = 0;

    // Parameters
    virtual std::vector<Parameter> getParameters() const = 0;
    virtual bool setParameter(const std::string& name, float value) = 0;
    virtual float getParameter(const std::string& name) const = 0;

    // State management
    virtual void setEnabled(bool enabled) { enabled_ = enabled; }
    virtual void setBypassed(bool bypassed) { bypassed_ = bypassed; }
    virtual bool isEnabled() const { return enabled_; }
    virtual bool isBypassed() const { return bypassed_; }

    // Identification
    virtual std::string getName() const { return name_; }
    virtual std::string getType() const = 0;
    virtual void setId(size_t id) { id_ = id; }
    virtual size_t getId() const { return id_; }

protected:
    std::string name_;
    size_t id_;
    double sampleRate_;
    int bufferSize_;
    int channels_;
    bool enabled_;
    bool bypassed_;
    mutable std::mutex parameterMutex_;
};

/**
 * @brief Gain processor with smooth parameter interpolation
 */
class GainProcessor : public AudioProcessor {
public:
    GainProcessor();
    ~GainProcessor() override = default;

    bool initialize(const std::string& name, double sampleRate, int bufferSize, int channels) override;
    void shutdown() override;
    void process(juce::dsp::ProcessContextReplacing<float>& context) override;
    void prepare(double sampleRate, int bufferSize, int channels) override;
    void reset() override;

    std::vector<Parameter> getParameters() const override;
    bool setParameter(const std::string& name, float value) override;
    float getParameter(const std::string& name) const override;
    std::string getType() const override { return "Gain"; }

private:
    juce::dsp::Gain<float> gain_;
    float gainDb_;
    std::atomic<float> targetGainDb_{0.0f};
};

/**
 * @brief Parametric equalizer with configurable bands
 */
class EqualizerProcessor : public AudioProcessor {
public:
    EqualizerProcessor();
    ~EqualizerProcessor() override = default;

    bool initialize(const std::string& name, double sampleRate, int bufferSize, int channels) override;
    void shutdown() override;
    void process(juce::dsp::ProcessContextReplacing<float>& context) override;
    void prepare(double sampleRate, int bufferSize, int channels) override;
    void reset() override;

    // Equalizer-specific configuration
    bool configure(const std::vector<double>& frequencies,
                   const std::vector<double>& gains,
                   const std::vector<double>& qValues);
    bool setBandGain(int band, double gainDb);
    double getBandGain(int band) const;
    int getNumBands() const { return numBands_; }

    std::vector<Parameter> getParameters() const override;
    bool setParameter(const std::string& name, float value) override;
    float getParameter(const std::string& name) const override;
    std::string getType() const override { return "Equalizer"; }

    bool supportsGPU() const override { return true; }
    void processGPU(float* audioData, size_t numSamples, GPUProcessor* gpuProcessor) override;

private:
    struct Band {
        double frequency;
        double gain;
        double q;
        juce::dsp::IIR::Filter<float> filter;
        juce::dsp::IIR::Coefficients<float>::Ptr coefficients;
    };

    std::vector<Band> bands_;
    int numBands_;
    std::atomic<bool> coefficientsChanged_{false};
    mutable std::mutex bandsMutex_;

    void updateCoefficients();
    std::string getBandParameterName(int band, const std::string& param) const;
};

/**
 * @brief Dynamics processor (compressor/limiter/expander)
 */
class DynamicsProcessor : public AudioProcessor {
public:
    enum class Type {
        Compressor,
        Limiter,
        Expander,
        Gate
    };

    DynamicsProcessor();
    ~DynamicsProcessor() override = default;

    bool initialize(const std::string& name, double sampleRate, int bufferSize, int channels) override;
    void shutdown() override;
    void process(juce::dsp::ProcessContextReplacing<float>& context) override;
    void prepare(double sampleRate, int bufferSize, int channels) override;
    void reset() override;

    void setType(Type type);
    Type getType() const { return type_; }

    std::vector<Parameter> getParameters() const override;
    bool setParameter(const std::string& name, float value) override;
    float getParameter(const std::string& name) const override;
    std::string getType() const override { return "Dynamics"; }

private:
    struct ChannelDynamics {
        juce::dsp::Compressor<float> compressor;
        juce::dsp::Limiter<float> limiter;
        float envelope = 0.0f;
    };

    std::vector<ChannelDynamics> channelDynamics_;
    Type type_;

    float thresholdDb_;
    float ratio_;
    float attackMs_;
    float releaseMs_;
    float kneeWidthDb_;
    float makeupGainDb_;
    float rangeDb_;

    void updateParameters();
};

/**
 * @brief Convolution processor for impulse response processing
 */
class ConvolutionProcessor : public AudioProcessor {
public:
    ConvolutionProcessor();
    ~ConvolutionProcessor() override = default;

    bool initialize(const std::string& name, double sampleRate, int bufferSize, int channels) override;
    void shutdown() override;
    void process(juce::dsp::ProcessContextReplacing<float>& context) override;
    void prepare(double sampleRate, int bufferSize, int channels) override;
    void reset() override;

    bool loadImpulseResponse(const std::vector<float>& ir, double irSampleRate);
    bool loadImpulseResponseFromFile(const std::string& filePath);
    void clearImpulseResponse();

    std::vector<Parameter> getParameters() const override;
    bool setParameter(const std::string& name, float value) override;
    float getParameter(const std::string& name) const override;
    std::string getType() const override { return "Convolution"; }

    bool supportsGPU() const override { return true; }
    void processGPU(float* audioData, size_t numSamples, GPUProcessor* gpuProcessor) override;

    size_t getIRLength() const { return irLength_; }
    double getIRSampleRate() const { return irSampleRate_; }

private:
    juce::dsp::Convolution convolution_;
    std::vector<float> impulseResponse_;
    size_t irLength_;
    double irSampleRate_;
    float wetLevel_;
    float dryLevel_;
    float latencyCompensationMs_;

    mutable std::mutex irMutex_;
    void updateConvolution();
};

} // namespace vortex