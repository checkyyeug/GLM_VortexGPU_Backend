#include "audio_processor.hpp"
#include "system/logger.hpp"

#include <algorithm>
#include <cmath>

namespace vortex {

// AudioProcessor base class
AudioProcessor::AudioProcessor(const std::string& name)
    : name_(name)
    , id_(0)
    , sampleRate_(44100.0)
    , bufferSize_(512)
    , channels_(2)
    , enabled_(true)
    , bypassed_(false)
{
}

// GainProcessor implementation
GainProcessor::GainProcessor()
    : AudioProcessor("Gain")
    , gainDb_(0.0f)
{
}

bool GainProcessor::initialize(const std::string& name, double sampleRate, int bufferSize, int channels) {
    name_ = name;
    sampleRate_ = sampleRate;
    bufferSize_ = bufferSize;
    channels_ = channels;

    gain_.setGainLinear(1.0f);
    gain_.setRampDurationSeconds(0.05); // 50ms ramp

    Logger::info("GainProcessor '{}' initialized", name_);
    return true;
}

void GainProcessor::shutdown() {
    Logger::info("GainProcessor '{}' shutdown", name_);
}

void GainProcessor::process(juce::dsp::ProcessContextReplacing<float>& context) {
    if (bypassed_) {
        return;
    }

    // Smooth gain changes
    float currentGain = juce::Decibels::decibelsToGain(targetGainDb_.load());
    gain_.setGainLinear(currentGain);

    gain_.process(context);
}

void GainProcessor::prepare(double sampleRate, int bufferSize, int channels) {
    sampleRate_ = sampleRate;
    bufferSize_ = bufferSize;
    channels_ = channels;

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = bufferSize;
    spec.numChannels = channels;

    gain_.prepare(spec);
}

void GainProcessor::reset() {
    gain_.reset();
    gainDb_ = 0.0f;
    targetGainDb_.store(0.0f);
}

std::vector<Parameter> GainProcessor::getParameters() const {
    std::lock_guard<std::mutex> lock(parameterMutex_);

    std::vector<Parameter> params;
    params.push_back({
        "gain_db",
        "Gain",
        "dB",
        gainDb_,
        0.0f,
        -60.0f,
        24.0f,
        0.1f,
        true,
        "Gain"
    });

    return params;
}

bool GainProcessor::setParameter(const std::string& name, float value) {
    if (name == "gain_db") {
        std::lock_guard<std::mutex> lock(parameterMutex_);
        gainDb_ = value;
        targetGainDb_.store(value);
        return true;
    }
    return false;
}

float GainProcessor::getParameter(const std::string& name) const {
    if (name == "gain_db") {
        std::lock_guard<std::mutex> lock(parameterMutex_);
        return gainDb_;
    }
    return 0.0f;
}

// EqualizerProcessor implementation
EqualizerProcessor::EqualizerProcessor()
    : AudioProcessor("Equalizer")
    , numBands_(10)
{
    bands_.reserve(10);
}

bool EqualizerProcessor::initialize(const std::string& name, double sampleRate, int bufferSize, int channels) {
    name_ = name;
    sampleRate_ = sampleRate;
    bufferSize_ = bufferSize;
    channels_ = channels;

    // Initialize default bands
    std::vector<double> frequencies = {
        31.5,   63,   125,  250,  500,
        1000,   2000, 4000, 8000,  16000
    };
    std::vector<double> gains(numBands_, 0.0);
    std::vector<double> q(numBands_, 1.414);

    return configure(frequencies, gains, q);
}

void EqualizerProcessor::shutdown() {
    std::lock_guard<std::mutex> lock(bandsMutex_);
    bands_.clear();
    Logger::info("EqualizerProcessor '{}' shutdown", name_);
}

void EqualizerProcessor::process(juce::dsp::ProcessContextReplacing<float>& context) {
    if (bypassed_) {
        return;
    }

    std::lock_guard<std::mutex> lock(bandsMutex_);

    // Update coefficients if they changed
    if (coefficientsChanged_.exchange(false)) {
        updateCoefficients();
    }

    // Process through all bands
    for (auto& band : bands_) {
        band.filter.process(context);
    }
}

void EqualizerProcessor::prepare(double sampleRate, int bufferSize, int channels) {
    sampleRate_ = sampleRate;
    bufferSize_ = bufferSize;
    channels_ = channels;

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = bufferSize;
    spec.numChannels = channels;

    std::lock_guard<std::mutex> lock(bandsMutex_);
    for (auto& band : bands_) {
        band.filter.prepare(spec);
    }

    updateCoefficients();
}

void EqualizerProcessor::reset() {
    std::lock_guard<std::mutex> lock(bandsMutex_);
    for (auto& band : bands_) {
        band.filter.reset();
    }
}

bool EqualizerProcessor::configure(const std::vector<double>& frequencies,
                                   const std::vector<double>& gains,
                                   const std::vector<double>& qValues) {
    if (frequencies.size() != gains.size() || frequencies.size() != qValues.size()) {
        Logger::error("Equalizer configuration: mismatched array sizes");
        return false;
    }

    std::lock_guard<std::mutex> lock(bandsMutex_);

    bands_.clear();
    numBands_ = static_cast<int>(frequencies.size());

    for (int i = 0; i < numBands_; ++i) {
        Band band;
        band.frequency = frequencies[i];
        band.gain = gains[i];
        band.q = qValues[i];
        band.coefficients = juce::dsp::IIR::Coefficients<float>::makePeakFilter(
            sampleRate_, band.frequency, band.q, juce::Decibels::decibelsToGain(band.gain));

        bands_.push_back(std::move(band));
    }

    coefficientsChanged_ = true;
    Logger::info("Equalizer configured with {} bands", numBands_);
    return true;
}

bool EqualizerProcessor::setBandGain(int band, double gainDb) {
    if (band < 0 || band >= numBands_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(bandsMutex_);
    bands_[band].gain = gainDb;
    coefficientsChanged_ = true;
    return true;
}

double EqualizerProcessor::getBandGain(int band) const {
    if (band < 0 || band >= numBands_) {
        return 0.0;
    }

    std::lock_guard<std::mutex> lock(bandsMutex_);
    return bands_[band].gain;
}

void EqualizerProcessor::updateCoefficients() {
    for (auto& band : bands_) {
        *band.coefficients = juce::dsp::IIR::Coefficients<float>::makePeakFilter(
            sampleRate_, band.frequency, band.q, juce::Decibels::decibelsToGain(band.gain));
    }
}

std::string EqualizerProcessor::getBandParameterName(int band, const std::string& param) const {
    return "band_" + std::to_string(band) + "_" + param;
}

std::vector<Parameter> EqualizerProcessor::getParameters() const {
    std::lock_guard<std::mutex> lock(bandsMutex_);

    std::vector<Parameter> params;
    params.reserve(numBands_);

    for (int i = 0; i < numBands_; ++i) {
        params.push_back({
            getBandParameterName(i, "gain"),
            "Band " + std::to_string(i + 1) + " Gain",
            "dB",
            static_cast<float>(bands_[i].gain),
            0.0f,
            -24.0f,
            24.0f,
            0.1f,
            true,
            "Equalizer"
        });
    }

    return params;
}

bool EqualizerProcessor::setParameter(const std::string& name, float value) {
    std::lock_guard<std::mutex> lock(parameterMutex_);

    // Parse band parameter name: "band_X_gain"
    if (name.substr(0, 5) == "band_") {
        size_t underscorePos = name.find('_', 5);
        if (underscorePos != std::string::npos) {
            int band = std::stoi(name.substr(5, underscorePos - 5));
            std::string param = name.substr(underscorePos + 1);

            if (param == "gain" && band >= 0 && band < numBands_) {
                std::lock_guard<std::mutex> bandLock(bandsMutex_);
                bands_[band].gain = value;
                coefficientsChanged_ = true;
                return true;
            }
        }
    }

    return false;
}

float EqualizerProcessor::getParameter(const std::string& name) const {
    std::lock_guard<std::mutex> lock(parameterMutex_);

    // Parse band parameter name
    if (name.substr(0, 5) == "band_") {
        size_t underscorePos = name.find('_', 5);
        if (underscorePos != std::string::npos) {
            int band = std::stoi(name.substr(5, underscorePos - 5));
            std::string param = name.substr(underscorePos + 1);

            if (param == "gain" && band >= 0 && band < numBands_) {
                std::lock_guard<std::mutex> bandLock(bandsMutex_);
                return static_cast<float>(bands_[band].gain);
            }
        }
    }

    return 0.0f;
}

void EqualizerProcessor::processGPU(float* audioData, size_t numSamples, GPUProcessor* gpuProcessor) {
    if (!gpuProcessor || bypassed_) {
        processCPU(audioData, numSamples);
        return;
    }

    // GPU implementation would go here
    // For now, fall back to CPU processing
    processCPU(audioData, numSamples);
}

// DynamicsProcessor implementation
DynamicsProcessor::DynamicsProcessor()
    : AudioProcessor("Dynamics")
    , type_(Type::Compressor)
    , thresholdDb_(-20.0f)
    , ratio_(4.0f)
    , attackMs_(5.0f)
    , releaseMs_(50.0f)
    , kneeWidthDb_(2.0f)
    , makeupGainDb_(0.0f)
    , rangeDb_(60.0f)
{
}

bool DynamicsProcessor::initialize(const std::string& name, double sampleRate, int bufferSize, int channels) {
    name_ = name;
    sampleRate_ = sampleRate;
    bufferSize_ = bufferSize;
    channels_ = channels;

    channelDynamics_.resize(channels);

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = bufferSize;
    spec.numChannels = channels;

    for (auto& dynamics : channelDynamics_) {
        dynamics.compressor.prepare(spec);
        dynamics.limiter.prepare(spec);
    }

    updateParameters();

    Logger::info("DynamicsProcessor '{}' initialized as {}", name_, static_cast<int>(type_));
    return true;
}

void DynamicsProcessor::shutdown() {
    channelDynamics_.clear();
    Logger::info("DynamicsProcessor '{}' shutdown", name_);
}

void DynamicsProcessor::process(juce::dsp::ProcessContextReplacing<float>& context) {
    if (bypassed_) {
        return;
    }

    auto& outputBlock = context.getOutputBlock();

    for (size_t channel = 0; channel < outputBlock.getNumChannels(); ++channel) {
        auto channelBlock = outputBlock.getSingleChannelBlock(channel);
        juce::dsp::ProcessContextReplacing<float> channelContext(channelBlock);

        switch (type_) {
            case Type::Compressor:
                channelDynamics_[channel].compressor.process(channelContext);
                break;
            case Type::Limiter:
                channelDynamics_[channel].limiter.process(channelContext);
                break;
            default:
                break;
        }
    }
}

void DynamicsProcessor::prepare(double sampleRate, int bufferSize, int channels) {
    sampleRate_ = sampleRate;
    bufferSize_ = bufferSize;
    channels_ = channels;

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = bufferSize;
    spec.numChannels = channels;

    channelDynamics_.resize(channels);
    for (auto& dynamics : channelDynamics_) {
        dynamics.compressor.prepare(spec);
        dynamics.limiter.prepare(spec);
    }

    updateParameters();
}

void DynamicsProcessor::reset() {
    for (auto& dynamics : channelDynamics_) {
        dynamics.compressor.reset();
        dynamics.limiter.reset();
        dynamics.envelope = 0.0f;
    }
}

void DynamicsProcessor::setType(Type type) {
    type_ = type;
    updateParameters();
}

void DynamicsProcessor::updateParameters() {
    for (auto& dynamics : channelDynamics_) {
        switch (type_) {
            case Type::Compressor:
                dynamics.compressor.setThreshold(thresholdDb_);
                dynamics.compressor.setRatio(ratio_);
                dynamics.compressor.setAttack(attackMs_ * 0.001);
                dynamics.compressor.setRelease(releaseMs_ * 0.001);
                break;
            case Type::Limiter:
                dynamics.limiter.setThreshold(thresholdDb_);
                dynamics.limiter.setRelease(releaseMs_ * 0.001);
                break;
            default:
                break;
        }
    }
}

std::vector<Parameter> DynamicsProcessor::getParameters() const {
    std::lock_guard<std::mutex> lock(parameterMutex_);

    std::vector<Parameter> params = {
        {"threshold_db", "Threshold", "dB", thresholdDb_, -20.0f, -60.0f, 0.0f, 0.1f, true, "Dynamics"},
        {"ratio", "Ratio", "", ratio_, 4.0f, 1.0f, 100.0f, 0.1f, true, "Dynamics"},
        {"attack_ms", "Attack", "ms", attackMs_, 5.0f, 0.1f, 1000.0f, 0.1f, true, "Dynamics"},
        {"release_ms", "Release", "ms", releaseMs_, 50.0f, 1.0f, 5000.0f, 0.1f, true, "Dynamics"},
        {"knee_width_db", "Knee Width", "dB", kneeWidthDb_, 2.0f, 0.0f, 10.0f, 0.1f, true, "Dynamics"},
        {"makeup_gain_db", "Makeup Gain", "dB", makeupGainDb_, 0.0f, -20.0f, 24.0f, 0.1f, true, "Dynamics"}
    };

    return params;
}

bool DynamicsProcessor::setParameter(const std::string& name, float value) {
    std::lock_guard<std::mutex> lock(parameterMutex_);

    if (name == "threshold_db") {
        thresholdDb_ = value;
    } else if (name == "ratio") {
        ratio_ = value;
    } else if (name == "attack_ms") {
        attackMs_ = value;
    } else if (name == "release_ms") {
        releaseMs_ = value;
    } else if (name == "knee_width_db") {
        kneeWidthDb_ = value;
    } else if (name == "makeup_gain_db") {
        makeupGainDb_ = value;
    } else {
        return false;
    }

    updateParameters();
    return true;
}

float DynamicsProcessor::getParameter(const std::string& name) const {
    std::lock_guard<std::mutex> lock(parameterMutex_);

    if (name == "threshold_db") return thresholdDb_;
    if (name == "ratio") return ratio_;
    if (name == "attack_ms") return attackMs_;
    if (name == "release_ms") return releaseMs_;
    if (name == "knee_width_db") return kneeWidthDb_;
    if (name == "makeup_gain_db") return makeupGainDb_;

    return 0.0f;
}

// ConvolutionProcessor implementation
ConvolutionProcessor::ConvolutionProcessor()
    : AudioProcessor("Convolution")
    , irLength_(0)
    , irSampleRate_(44100.0)
    , wetLevel_(1.0f)
    , dryLevel_(0.0f)
    , latencyCompensationMs_(0.0f)
{
}

bool ConvolutionProcessor::initialize(const std::string& name, double sampleRate, int bufferSize, int channels) {
    name_ = name;
    sampleRate_ = sampleRate;
    bufferSize_ = bufferSize;
    channels_ = channels;

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = bufferSize;
    spec.numChannels = channels;

    convolution_.prepare(spec);

    Logger::info("ConvolutionProcessor '{}' initialized", name_);
    return true;
}

void ConvolutionProcessor::shutdown() {
    convolution_.reset();
    std::lock_guard<std::mutex> lock(irMutex_);
    impulseResponse_.clear();
    irLength_ = 0;
    Logger::info("ConvolutionProcessor '{}' shutdown", name_);
}

void ConvolutionProcessor::process(juce::dsp::ProcessContextReplacing<float>& context) {
    if (bypassed_ || impulseResponse_.empty()) {
        return;
    }

    convolution_.process(context);
}

void ConvolutionProcessor::prepare(double sampleRate, int bufferSize, int channels) {
    sampleRate_ = sampleRate;
    bufferSize_ = bufferSize;
    channels_ = channels;

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = bufferSize;
    spec.numChannels = channels;

    convolution_.prepare(spec);
    updateConvolution();
}

void ConvolutionProcessor::reset() {
    convolution_.reset();
}

bool ConvolutionProcessor::loadImpulseResponse(const std::vector<float>& ir, double irSampleRate) {
    if (ir.empty()) {
        Logger::warning("Empty impulse response provided");
        return false;
    }

    std::lock_guard<std::mutex> lock(irMutex_);
    impulseResponse_ = ir;
    irLength_ = ir.size();
    irSampleRate_ = irSampleRate;

    updateConvolution();

    Logger::info("Loaded impulse response: {} samples at {} Hz", irLength_, irSampleRate_);
    return true;
}

bool ConvolutionProcessor::loadImpulseResponseFromFile(const std::string& filePath) {
    // Implementation would use JUCE's AudioFormatReader to load IR
    // For now, return false as placeholder
    Logger::warning("IR file loading not yet implemented: {}", filePath);
    return false;
}

void ConvolutionProcessor::clearImpulseResponse() {
    std::lock_guard<std::mutex> lock(irMutex_);
    impulseResponse_.clear();
    irLength_ = 0;
    convolution_.loadImpulseResponse({}, static_cast<double>(sampleRate_), 0, 0);
}

void ConvolutionProcessor::updateConvolution() {
    std::lock_guard<std::mutex> lock(irMutex_);
    if (!impulseResponse_.empty()) {
        convolution_.loadImpulseResponse(
            juce::AudioBuffer<float>(1, static_cast<int>(irLength_)),
            irSampleRate_, 0, 0);
    }
}

std::vector<Parameter> ConvolutionProcessor::getParameters() const {
    std::lock_guard<std::mutex> lock(parameterMutex_);

    return {
        {"wet_level", "Wet Level", "", wetLevel_, 1.0f, 0.0f, 2.0f, 0.01f, true, "Convolution"},
        {"dry_level", "Dry Level", "", dryLevel_, 0.0f, 0.0f, 2.0f, 0.01f, true, "Convolution"},
        {"latency_compensation_ms", "Latency Compensation", "ms", latencyCompensationMs_, 0.0f, -100.0f, 100.0f, 0.1f, true, "Convolution"}
    };
}

bool ConvolutionProcessor::setParameter(const std::string& name, float value) {
    std::lock_guard<std::mutex> lock(parameterMutex_);

    if (name == "wet_level") {
        wetLevel_ = value;
    } else if (name == "dry_level") {
        dryLevel_ = value;
    } else if (name == "latency_compensation_ms") {
        latencyCompensationMs_ = value;
    } else {
        return false;
    }

    updateConvolution();
    return true;
}

float ConvolutionProcessor::getParameter(const std::string& name) const {
    std::lock_guard<std::mutex> lock(parameterMutex_);

    if (name == "wet_level") return wetLevel_;
    if (name == "dry_level") return dryLevel_;
    if (name == "latency_compensation_ms") return latencyCompensationMs_;

    return 0.0f;
}

void ConvolutionProcessor::processGPU(float* audioData, size_t numSamples, GPUProcessor* gpuProcessor) {
    if (!gpuProcessor || bypassed_ || impulseResponse_.empty()) {
        processCPU(audioData, numSamples);
        return;
    }

    // GPU implementation would go here
    // For now, fall back to CPU processing
    processCPU(audioData, numSamples);
}

} // namespace vortex