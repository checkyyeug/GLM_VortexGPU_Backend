#include "core/dsp/time_domain_effects.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <random>
#include <immintrin.h>

namespace vortex::core::dsp {

// DelayLine implementation
DelayLine::DelayLine()
    : max_delay_samples_(0)
    , channels_(0)
    , delay_line_size_(0)
    , interpolation_type_(InterpolationType::LINEAR)
    , feedback_(0.0f)
    , wet_mix_(1.0f)
    , filter_enabled_(false)
    , filter_frequency_(20000.0f)
    , filter_resonance_(0.7f) {
}

DelayLine::~DelayLine() {
    shutdown();
}

bool DelayLine::initialize(uint32_t max_delay_samples, uint32_t channels, InterpolationType interpolation) {
    if (max_delay_samples == 0 || channels == 0) {
        Logger::error("Invalid delay line parameters");
        return false;
    }

    // Initialize delay buffers
    delay_buffers_.resize(channels);
    for (auto& buffer : delay_buffers_) {
        buffer.resize(max_delay_samples, 0.0f);
    }

    // Initialize write positions
    write_positions_.resize(channels, 0);

    // Initialize filter states
    filter_states_.resize(channels, std::vector<float>(2, 0.0f));

    max_delay_samples_ = max_delay_samples;
    channels_ = channels;
    delay_line_size_ = max_delay_samples;
    interpolation_type_ = interpolation;

    // Initialize filter coefficients
    filter_coefficients_.resize(5, 0.0f);
    update_filter_coefficients();

    Logger::info("Initialized delay line: {} samples, {} channels, interpolation type {}",
                max_delay_samples_, channels_, static_cast<int>(interpolation));

    return true;
}

void DelayLine::shutdown() {
    delay_buffers_.clear();
    write_positions_.clear();
    filter_states_.clear();
    filter_coefficients_.clear();

    max_delay_samples_ = 0;
    channels_ = 0;
    delay_line_size_ = 0;
}

void DelayLine::clear() {
    for (auto& buffer : delay_buffers_) {
        std::fill(buffer.begin(), buffer.end(), 0.0f);
    }

    std::fill(write_positions_.begin(), write_positions_.end(), 0);

    for (auto& state : filter_states_) {
        std::fill(state.begin(), state.end(), 0.0f);
    }
}

float DelayLine::read_sample(uint32_t channel, float delay_samples) const {
    if (channel >= channels_ || delay_samples >= max_delay_samples_) {
        return 0.0f;
    }

    uint32_t write_pos = write_positions_[channel];
    float read_pos = write_pos - delay_samples;
    if (read_pos < 0.0f) {
        read_pos += delay_line_size_;
    }

    switch (interpolation_type_) {
        case InterpolationType::NONE:
            return delay_buffers_[channel][static_cast<uint32_t>(read_pos)];

        case InterpolationType::LINEAR:
            return interpolate_linear(channel, read_pos);

        case InterpolationType::CUBIC:
            return interpolate_cubic(channel, read_pos);

        case InterpolationType::HERMITE:
            return interpolate_hermite(channel, read_pos);

        case InterpolationType::THIRAN:
            return interpolate_thiran(channel, delay_samples);

        default:
            return interpolate_linear(channel, read_pos);
    }
}

void DelayLine::write_sample(uint32_t channel, float sample) {
    if (channel >= channels_) {
        return;
    }

    // Apply filter if enabled
    if (filter_enabled_) {
        process_filter(channel, sample);
    }

    // Apply feedback
    float feedback_sample = sample + read_sample(channel, static_cast<float>(delay_line_size_ * feedback_));

    delay_buffers_[channel][write_positions_[channel]] = feedback_sample;
    write_positions_[channel] = (write_positions_[channel] + 1) % delay_line_size_;
}

void DelayLine::process_block(const float* input, float* output, uint32_t frame_count,
                             const std::vector<float>& delay_times) {
    if (!input || !output || frame_count == 0) {
        return;
    }

    for (uint32_t frame = 0; frame < frame_count; ++frame) {
        for (uint32_t ch = 0; ch < channels_; ++ch) {
            uint32_t input_index = frame * channels_ + ch;
            float delay_samples = delay_times.empty() ?
                max_delay_samples_ / 2 : time_domain_utils::milliseconds_to_samples(delay_times[ch % delay_times.size()], 44100);

            float wet_sample = read_sample(ch, delay_samples);
            write_sample(ch, input[input_index]);

            uint32_t output_index = frame * channels_ + ch;
            output[output_index] = wet_sample * wet_mix_ + input[input_index] * (1.0f - wet_mix_);
        }
    }
}

void DelayLine::set_max_delay_samples(uint32_t max_delay_samples) {
    if (max_delay_samples > max_delay_samples_) {
        max_delay_samples_ = max_delay_samples;
        delay_line_size_ = max_delay_samples;

        for (auto& buffer : delay_buffers_) {
            buffer.resize(max_delay_samples, 0.0f);
        }
    }
}

void DelayLine::set_interpolation_type(InterpolationType type) {
    interpolation_type_ = type;
}

void DelayLine::set_feedback(float feedback) {
    feedback_ = std::clamp(feedback, -0.99f, 0.99f);
}

void DelayLine::set_wet_mix(float mix) {
    wet_mix_ = std::clamp(mix, 0.0f, 1.0f);
}

void DelayLine::set_filter_enabled(bool enabled) {
    filter_enabled_ = enabled;
}

void DelayLine::set_filter_parameters(float frequency, float resonance) {
    filter_frequency_ = frequency;
    filter_resonance_ = std::clamp(resonance, 0.1f, 10.0f);
    update_filter_coefficients();
}

float DelayLine::interpolate_linear(uint32_t channel, float read_pos) const {
    uint32_t pos1 = static_cast<uint32_t>(read_pos);
    uint32_t pos2 = (pos1 + 1) % delay_line_size_;
    float fraction = read_pos - static_cast<float>(pos1);

    return delay_buffers_[channel][pos1] * (1.0f - fraction) +
           delay_buffers_[channel][pos2] * fraction;
}

float DelayLine::interpolate_cubic(uint32_t channel, float read_pos) const {
    uint32_t pos0 = static_cast<uint32_t>(read_pos) - 1;
    uint32_t pos1 = static_cast<uint32_t>(read_pos);
    uint32_t pos2 = (pos1 + 1) % delay_line_size_;
    uint32_t pos3 = (pos2 + 1) % delay_line_size_;
    float fraction = read_pos - static_cast<float>(pos1);

    float a0 = delay_buffers_[channel][pos0];
    float a1 = delay_buffers_[channel][pos1];
    float a2 = delay_buffers_[channel][pos2];
    float a3 = delay_buffers_[channel][pos3];

    return time_domain_utils::cubic_interpolate(a0, a1, a2, a3, fraction);
}

float DelayLine::interpolate_hermite(uint32_t channel, float read_pos) const {
    uint32_t pos0 = static_cast<uint32_t>(read_pos) - 1;
    uint32_t pos1 = static_cast<uint32_t>(read_pos);
    uint32_t pos2 = (pos1 + 1) % delay_line_size_;
    uint32_t pos3 = (pos2 + 1) % delay_line_size_;
    float fraction = read_pos - static_cast<float>(pos1);

    float a0 = delay_buffers_[channel][pos0];
    float a1 = delay_buffers_[channel][pos1];
    float a2 = delay_buffers_[channel][pos2];
    float a3 = delay_buffers_[channel][pos3];

    return time_domain_utils::hermite_interpolate(a0, a1, a2, a3, fraction);
}

float DelayLine::interpolate_thiran(uint32_t channel, float delay_samples) const {
    // Thiran all-pass interpolation for fractional delays
    float fractional_delay = delay_samples - std::floor(delay_samples);
    float delay_fraction = 1.0f - fractional_delay;

    if (delay_fraction < 0.001f) {
        return read_sample(channel, delay_samples);
    }

    float a1 = (delay_fraction - 1.0f) / (delay_fraction + 1.0f);
    uint32_t write_pos = write_positions_[channel];
    float read_pos = write_pos - delay_samples;

    if (read_pos < 0.0f) {
        read_pos += delay_line_size_;
    }

    uint32_t pos1 = static_cast<uint32_t>(read_pos);
    uint32_t pos2 = (pos1 + 1) % delay_line_size_;

    float current_input = delay_buffers_[channel][pos1];
    float previous_output = filter_states_[channel][0]; // Use filter state for previous output

    float output = a1 * (current_input - previous_output) + current_input;

    // Store for next iteration
    filter_states_[channel][0] = output;

    return output;
}

void DelayLine::process_filter(uint32_t channel, float& sample) {
    if (!filter_enabled_) {
        return;
    }

    float& x1 = filter_states_[channel][0];
    float& x2 = filter_states_[channel][1];

    float output = filter_coefficients_[0] * sample +
                  filter_coefficients_[1] * x1 +
                  filter_coefficients_[2] * x2 +
                  filter_coefficients_[3] * x1 + // b1 * x1 (previous output)
                  filter_coefficients_[4] * x2; // b2 * x2 (previous output)

    x2 = x1;
    x1 = sample;

    sample = output;
}

void DelayLine::update_filter_coefficients() {
    if (!filter_enabled_) {
        return;
    }

    // For simplicity, implementing low-pass filter
    // In a real implementation, would have configurable filter types
    time_domain_utils::calculate_low_pass_coefficients(
        filter_frequency_, 44100.0f, filter_resonance_,
        filter_coefficients_[0], filter_coefficients_[1], filter_coefficients_[2],
        filter_coefficients_[3], filter_coefficients_[4]);
}

// MultiTapDelayProcessor implementation
MultiTapDelayProcessor::MultiTapDelayProcessor()
    : modulation_type_(TimeDomainEffectType::MULTI_TAP_DELAY)
    , bypassed_(false)
    , dry_wet_mix_(0.3f)
    , sample_rate_(44100)
    , channels_(2)
    , max_frame_size_(4096) {
}

MultiTapDelayProcessor::~MultiTapDelayProcessor() {
    shutdown();
}

bool MultiTapDelayProcessor::initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    sample_rate_ = sample_rate;
    channels_ = channels;
    max_frame_size_ = max_frame_size;

    Logger::info("Initializing Multi-Tap Delay Processor: {} Hz, {} channels, max frame: {}",
                sample_rate, channels, max_frame_size);

    // Initialize parameters
    parameters_ = TimeDomainParameters();
    parameters_.num_taps = 3;
    parameters_.tap_delay_times_ms = {250.0f, 500.0f, 750.0f};
    parameters_.tap_gain_levels = {1.0f, 0.7f, 0.5f};
    parameters_.tap_feedback_levels = {0.3f, 0.2f, 0.1f};

    // Initialize taps
    update_tap_parameters();

    // Initialize processing buffers
    uint32_t total_samples = max_frame_size * channels;
    wet_buffer_.resize(total_samples, 0.0f);
    dry_buffer_.resize(total_samples, 0.0f);
    tap_buffer_.resize(total_samples, 0.0f);

    // Reset statistics
    statistics_ = TimeDomainStatistics();
    statistics_.delay_line_size_samples = taps_.empty() ? 0 : taps_[0].delay_line->get_delay_line_size();

    // Initialize some presets
    presets_["default"] = parameters_;
    presets_["slapback"] = parameters_;
    presets_["slapback"].tap_delay_times_ms = {120.0f};
    presets_["slapback"].tap_gain_levels = {0.8f};
    presets_["slapback"].tap_feedback_levels = {0.2f};

    Logger::info("Multi-Tap Delay Processor initialized with {} taps", taps_.size());
    return true;
}

void MultiTapDelayProcessor::shutdown() {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    taps_.clear();
    wet_buffer_.clear();
    dry_buffer_.clear();
    tap_buffer_.clear();
    presets_.clear();

    Logger::info("Multi-Tap Delay Processor shutdown");
}

bool MultiTapDelayProcessor::reset() {
    for (auto& tap : taps_) {
        if (tap.delay_line) {
            tap.delay_line->clear();
        }
    }

    std::fill(wet_buffer_.begin(), wet_buffer_.end(), 0.0f);
    std::fill(dry_buffer_.begin(), dry_buffer_.end(), 0.0f);
    std::fill(tap_buffer_.begin(), tap_buffer_.end(), 0.0f);

    Logger::info("Multi-Tap Delay Processor reset");
    return true;
}

bool MultiTapDelayProcessor::process(const float* input, float* output, uint32_t frame_count) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (bypassed_ || !input || !output || frame_count == 0) {
        if (input && output && frame_count > 0) {
            std::memcpy(output, input, frame_count * channels_ * sizeof(float));
        }
        return true;
    }

    // Store dry signal
    std::memcpy(dry_buffer_.data(), input, frame_count * channels_ * sizeof(float));

    // Clear wet buffer
    std::fill(wet_buffer_.begin(), wet_buffer_.begin() + frame_count * channels_, 0.0f);

    // Process each tap
    process_taps(input, wet_buffer_.data(), frame_count);

    // Apply dry/wet mix
    apply_dry_wet_mix(dry_buffer_.data(), wet_buffer_.data(), output, frame_count);

    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::lock_guard<std::mutex> lock(stats_mutex_);
    statistics_.total_process_calls++;
    statistics_.successful_calls++;

    double processing_time_us = duration.count();
    statistics_.avg_processing_time_us =
        ((statistics_.avg_processing_time_us * (statistics_.total_process_calls - 1)) + processing_time_us) /
        statistics_.total_process_calls;
    statistics_.max_processing_time_us = std::max(statistics_.max_processing_time_us, processing_time_us);
    statistics_.min_processing_time_us = std::min(statistics_.min_processing_time_us, processing_time_us);
    statistics_.is_active = true;

    return true;
}

bool MultiTapDelayProcessor::process_interleaved(const float* input, float* output, uint32_t frame_count) {
    // For this implementation, interleaved and non-interleaved processing are the same
    return process(input, output, frame_count);
}

bool MultiTapDelayProcessor::set_parameters(const TimeDomainParameters& params) {
    parameters_ = params;
    update_tap_parameters();

    Logger::info("Updated multi-tap delay parameters: {} taps", taps_.size());
    return true;
}

TimeDomainParameters MultiTapDelayProcessor::get_parameters() const {
    return parameters_;
}

bool MultiTapDelayProcessor::set_parameter(const std::string& name, float value) {
    if (name == "delay_time_ms") {
        parameters_.delay_time_ms = value;
        if (!parameters_.tap_delay_times_ms.empty()) {
            parameters_.tap_delay_times_ms[0] = value;
        }
    } else if (name == "feedback_percent") {
        parameters_.feedback_percent = value;
        if (!parameters_.tap_feedback_levels.empty()) {
            parameters_.tap_feedback_levels[0] = value / 100.0f;
        }
    } else if (name == "wet_mix_percent") {
        dry_wet_mix_ = value / 100.0f;
    } else {
        return false;
    }

    update_tap_parameters();
    return true;
}

float MultiTapDelayProcessor::get_parameter(const std::string& name) const {
    if (name == "delay_time_ms") {
        return parameters_.delay_time_ms;
    } else if (name == "feedback_percent") {
        return parameters_.feedback_percent;
    } else if (name == "wet_mix_percent") {
        return dry_wet_mix_ * 100.0f;
    }
    return 0.0f;
}

bool MultiTapDelayProcessor::set_bypass(bool bypass) {
    bypassed_ = bypass;
    Logger::info("Multi-tap delay bypass {}", bypass ? "enabled" : "disabled");
    return true;
}

bool MultiTapDelayProcessor::is_bypassed() const {
    return bypassed_;
}

bool MultiTapDelayProcessor::set_dry_wet_mix(float mix) {
    dry_wet_mix_ = std::clamp(mix, 0.0f, 1.0f);
    return true;
}

bool MultiTapDelayProcessor::save_preset(const std::string& name) {
    presets_[name] = parameters_;
    Logger::info("Saved preset: {}", name);
    return true;
}

bool MultiTapDelayProcessor::load_preset(const std::string& name) {
    auto it = presets_.find(name);
    if (it != presets_.end()) {
        parameters_ = it->second;
        update_tap_parameters();
        Logger::info("Loaded preset: {}", name);
        return true;
    }
    Logger::warn("Preset not found: {}", name);
    return false;
}

std::vector<std::string> MultiTapDelayProcessor::get_available_presets() const {
    std::vector<std::string> preset_names;
    for (const auto& preset : presets_) {
        preset_names.push_back(preset.first);
    }
    return preset_names;
}

TimeDomainEffectType MultiTapDelayProcessor::get_type() const {
    return modulation_type_;
}

std::string MultiTapDelayProcessor::get_name() const {
    return "Multi-Tap Delay";
}

std::string MultiTapDelayProcessor::get_version() const {
    return "1.0.0";
}

std::string MultiTapDelayProcessor::get_description() const {
    return "Professional multi-tap delay effect with configurable taps and feedback";
}

TimeDomainStatistics MultiTapDelayProcessor::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return statistics_;
}

void MultiTapDelayProcessor::reset_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    statistics_ = TimeDomainStatistics();
    statistics_.last_reset_time = std::chrono::steady_clock::now();
}

bool MultiTapDelayProcessor::supports_real_time_parameter_changes() const {
    return true;
}

bool MultiTapDelayProcessor::supports_gpu_acceleration() const {
    return false; // Time-domain effects typically benefit more from CPU optimization
}

uint32_t MultiTapDelayProcessor::get_delay_samples() const {
    if (taps_.empty()) {
        return 0;
    }
    return static_cast<uint32_t>(parameters_.tap_delay_times_ms[0] * sample_rate_ / 1000.0f);
}

double MultiTapDelayProcessor::get_expected_latency_ms() const {
    return 1.0; // ~1ms expected latency for real-time processing
}

void MultiTapDelayProcessor::set_num_taps(uint32_t num_taps) {
    parameters_.num_taps = std::clamp(num_taps, 1u, 16u);

    // Resize tap parameter arrays
    parameters_.tap_delay_times_ms.resize(parameters_.num_taps, 250.0f);
    parameters_.tap_gain_levels.resize(parameters_.num_taps, 1.0f);
    parameters_.tap_feedback_levels.resize(parameters_.num_taps, 0.3f);

    update_tap_parameters();
}

void MultiTapDelayProcessor::set_tap_delay(uint32_t tap_index, float delay_ms) {
    if (tap_index < parameters_.tap_delay_times_ms.size()) {
        parameters_.tap_delay_times_ms[tap_index] = delay_ms;
        update_tap_parameters();
    }
}

void MultiTapDelayProcessor::set_tap_gain(uint32_t tap_index, float gain) {
    if (tap_index < parameters_.tap_gain_levels.size()) {
        parameters_.tap_gain_levels[tap_index] = gain;
        update_tap_parameters();
    }
}

void MultiTapDelayProcessor::set_tap_feedback(uint32_t tap_index, float feedback) {
    if (tap_index < parameters_.tap_feedback_levels.size()) {
        parameters_.tap_feedback_levels[tap_index] = feedback;
        update_tap_parameters();
    }
}

void MultiTapDelayProcessor::update_tap_parameters() {
    // Resize taps array if needed
    while (taps_.size() < parameters_.num_taps) {
        taps_.emplace_back();
    }
    while (taps_.size() > parameters_.num_taps) {
        taps_.pop_back();
    }

    // Update each tap
    float max_delay_ms = 0.0f;
    for (size_t i = 0; i < taps_.size(); ++i) {
        auto& tap = taps_[i];

        tap.delay_time_ms = parameters_.tap_delay_times_ms[i];
        tap.gain = parameters_.tap_gain_levels[i];
        tap.feedback = parameters_.tap_feedback_levels[i];
        tap.delay_samples = static_cast<uint32_t>(tap.delay_time_ms * sample_rate_ / 1000.0f);

        max_delay_ms = std::max(max_delay_ms, tap.delay_time_ms);

        // Create or update delay line
        uint32_t max_delay_samples = static_cast<uint32_t>(max_delay_ms * sample_rate_ / 1000.0f) + 1024;
        if (!tap.delay_line) {
            tap.delay_line = std::make_unique<DelayLine>();
            tap.delay_line->initialize(max_delay_samples, channels_, InterpolationType::CUBIC);
        }

        tap.delay_line->set_feedback(tap.feedback);
        tap.delay_line->set_wet_mix(tap.gain);
    }

    std::lock_guard<std::mutex> lock(stats_mutex_);
    statistics_.delay_line_size_samples = taps_.empty() ? 0 :
        static_cast<uint32_t>(max_delay_ms * sample_rate_ / 1000.0f);
}

void MultiTapDelayProcessor::process_taps(const float* input, float* output, uint32_t frame_count) {
    std::fill(tap_buffer_.begin(), tap_buffer_.begin() + frame_count * channels_, 0.0f);

    for (size_t tap_idx = 0; tap_idx < taps_.size(); ++tap_idx) {
        auto& tap = taps_[tap_idx];

        if (!tap.delay_line) {
            continue;
        }

        // Process this tap
        tap.delay_line->process_block(input, tap_buffer_.data(), frame_count, {tap.delay_time_ms});

        // Add to wet buffer (mix all taps)
        for (uint32_t i = 0; i < frame_count * channels_; ++i) {
            output[i] += tap_buffer_[i];
        }
    }
}

void MultiTapDelayProcessor::apply_dry_wet_mix(const float* dry, const float* wet, float* output, uint32_t frame_count) {
    float wet_gain = dry_wet_mix_;
    float dry_gain = 1.0f - wet_gain;

    for (uint32_t i = 0; i < frame_count * channels_; ++i) {
        output[i] = dry[i] * dry_gain + wet[i] * wet_gain;
    }
}

// ReverbProcessor implementation (simplified)
ReverbProcessor::ReverbProcessor()
    : current_algorithm_(ReverbAlgorithm::FREEVERB)
    , bypassed_(false)
    , dry_wet_mix_(0.3f)
    , sample_rate_(44100)
    , channels_(2)
    , max_frame_size_(4096)
    , ir_length_(0) {
}

ReverbProcessor::~ReverbProcessor() {
    shutdown();
}

bool ReverbProcessor::initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    sample_rate_ = sample_rate;
    channels_ = channels;
    max_frame_size_ = max_frame_size;

    Logger::info("Initializing Reverb Processor: {} Hz, {} channels", sample_rate, channels);

    // Initialize parameters
    parameters_ = TimeDomainParameters();
    parameters_.reverb_algorithm = ReverbAlgorithm::FREEVERB;
    parameters_.room_size = 0.5f;
    parameters_.damping = 0.5f;
    parameters_.width = 1.0f;
    parameters_.predelay_ms = 20.0f;
    parameters_.diffusion = 0.7f;

    // Initialize reverb algorithm
    initialize_reverb_algorithm();

    // Initialize processing buffers
    uint32_t total_samples = max_frame_size * channels;
    wet_buffer_.resize(total_samples, 0.0f);
    dry_buffer_.resize(total_samples, 0.0f);
    temp_buffer_.resize(total_samples, 0.0f);

    // Reset statistics
    statistics_ = TimeDomainStatistics();

    // Initialize presets
    presets_["hall"] = parameters_;
    presets_["hall"].room_size = 0.8f;
    presets_["hall"].damping = 0.4f;

    presets_["room"] = parameters_;
    presets_["room"].room_size = 0.3f;
    presets_["room"].damping = 0.7f;

    Logger::info("Reverb Processor initialized with {} algorithm", static_cast<int>(current_algorithm_));
    return true;
}

void ReverbProcessor::shutdown() {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    comb_delay_lines_.clear();
    allpass_delay_lines_.clear();
    impulse_response_.clear();
    convolution_buffers_.clear();
    wet_buffer_.clear();
    dry_buffer_.clear();
    temp_buffer_.clear();
    presets_.clear();

    Logger::info("Reverb Processor shutdown");
}

bool ReverbProcessor::reset() {
    for (auto& delay_line : comb_delay_lines_) {
        if (delay_line) {
            delay_line->clear();
        }
    }

    for (auto& delay_line : allpass_delay_lines_) {
        if (delay_line) {
            delay_line->clear();
        }
    }

    std::fill(wet_buffer_.begin(), wet_buffer_.end(), 0.0f);
    std::fill(dry_buffer_.begin(), dry_buffer_.end(), 0.0f);
    std::fill(temp_buffer_.begin(), temp_buffer_.end(), 0.0f);

    Logger::info("Reverb Processor reset");
    return true;
}

bool ReverbProcessor::process(const float* input, float* output, uint32_t frame_count) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (bypassed_ || !input || !output || frame_count == 0) {
        if (input && output && frame_count > 0) {
            std::memcpy(output, input, frame_count * channels_ * sizeof(float));
        }
        return true;
    }

    // Store dry signal
    std::memcpy(dry_buffer_.data(), input, frame_count * channels_ * sizeof(float));

    // Clear wet buffer
    std::fill(wet_buffer_.begin(), wet_buffer_.begin() + frame_count * channels_, 0.0f);

    // Process reverb based on algorithm
    switch (current_algorithm_) {
        case ReverbAlgorithm::FREEVERB:
            process_freeverb(input, wet_buffer_.data(), frame_count);
            break;
        case ReverbAlgorithm::SCHROEDER:
            process_schroeder(input, wet_buffer_.data(), frame_count);
            break;
        case ReverbAlgorithm::CONVOLUTION:
            process_convolution(input, wet_buffer_.data(), frame_count);
            break;
        default:
            process_freeverb(input, wet_buffer_.data(), frame_count);
            break;
    }

    // Apply dry/wet mix
    float wet_gain = dry_wet_mix_;
    float dry_gain = 1.0f - wet_gain;
    for (uint32_t i = 0; i < frame_count * channels_; ++i) {
        output[i] = dry_buffer_[i] * dry_gain + wet_buffer_[i] * wet_gain;
    }

    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::lock_guard<std::mutex> lock(stats_mutex_);
    statistics_.total_process_calls++;
    statistics_.successful_calls++;

    double processing_time_us = duration.count();
    statistics_.avg_processing_time_us =
        ((statistics_.avg_processing_time_us * (statistics_.total_process_calls - 1)) + processing_time_us) /
        statistics_.total_process_calls;
    statistics_.max_processing_time_us = std::max(statistics_.max_processing_time_us, processing_time_us);
    statistics_.min_processing_time_us = std::min(statistics_.min_processing_time_us, processing_time_us);
    statistics_.is_active = true;

    return true;
}

bool ReverbProcessor::process_interleaved(const float* input, float* output, uint32_t frame_count) {
    return process(input, output, frame_count);
}

bool ReverbProcessor::set_parameters(const TimeDomainParameters& params) {
    parameters_ = params;
    current_algorithm_ = params.reverb_algorithm;
    update_algorithm_parameters();
    return true;
}

TimeDomainParameters ReverbProcessor::get_parameters() const {
    return parameters_;
}

bool ReverbProcessor::set_parameter(const std::string& name, float value) {
    if (name == "room_size") {
        parameters_.room_size = value;
    } else if (name == "damping") {
        parameters_.damping = value;
    } else if (name == "width") {
        parameters_.width = value;
    } else if (name == "wet_mix_percent") {
        dry_wet_mix_ = value / 100.0f;
    } else {
        return false;
    }

    update_algorithm_parameters();
    return true;
}

float ReverbProcessor::get_parameter(const std::string& name) const {
    if (name == "room_size") {
        return parameters_.room_size;
    } else if (name == "damping") {
        return parameters_.damping;
    } else if (name == "width") {
        return parameters_.width;
    } else if (name == "wet_mix_percent") {
        return dry_wet_mix_ * 100.0f;
    }
    return 0.0f;
}

bool ReverbProcessor::set_bypass(bool bypass) {
    bypassed_ = bypass;
    return true;
}

bool ReverbProcessor::is_bypassed() const {
    return bypassed_;
}

bool ReverbProcessor::set_dry_wet_mix(float mix) {
    dry_wet_mix_ = std::clamp(mix, 0.0f, 1.0f);
    return true;
}

TimeDomainEffectType ReverbProcessor::get_type() const {
    return TimeDomainEffectType::REVERB;
}

std::string ReverbProcessor::get_name() const {
    return "Reverb";
}

std::string ReverbProcessor::get_version() const {
    return "1.0.0";
}

std::string ReverbProcessor::get_description() const {
    return "Professional reverb effect with multiple algorithms";
}

TimeDomainStatistics ReverbProcessor::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return statistics_;
}

void ReverbProcessor::reset_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    statistics_ = TimeDomainStatistics();
    statistics_.last_reset_time = std::chrono::steady_clock::now();
}

bool ReverbProcessor::supports_real_time_parameter_changes() const {
    return true;
}

bool ReverbProcessor::supports_gpu_acceleration() const {
    return current_algorithm_ == ReverbAlgorithm::CONVOLUTION;
}

uint32_t ReverbProcessor::get_delay_samples() const {
    return static_cast<uint32_t>(parameters_.predelay_ms * sample_rate_ / 1000.0f);
}

double ReverbProcessor::get_expected_latency_ms() const {
    return 2.0; // Higher latency for reverb processing
}

void ReverbProcessor::initialize_reverb_algorithm() {
    switch (current_algorithm_) {
        case ReverbAlgorithm::FREEVERB: {
            // Create comb filter delay lines
            std::vector<float> comb_tuning;
            time_domain_utils::calculate_freeverb_comb_tuning(sample_rate_, comb_tuning);

            comb_delay_lines_.clear();
            for (float delay : comb_tuning) {
                auto delay_line = std::make_unique<DelayLine>();
                uint32_t delay_samples = static_cast<uint32_t>(delay);
                delay_line->initialize(delay_samples + 1024, channels_, InterpolationType::LINEAR);
                comb_delay_lines_.push_back(std::move(delay_line));
            }

            // Create all-pass filter delay lines
            std::vector<float> allpass_tuning;
            time_domain_utils::calculate_freeverb_allpass_tuning(sample_rate_, allpass_tuning);

            allpass_delay_lines_.clear();
            for (float delay : allpass_tuning) {
                auto delay_line = std::make_unique<DelayLine>();
                uint32_t delay_samples = static_cast<uint32_t>(delay);
                delay_line->initialize(delay_samples + 1024, channels_, InterpolationType::LINEAR);
                allpass_delay_lines_.push_back(std::move(delay_line));
            }
            break;
        }

        case ReverbAlgorithm::CONVOLUTION: {
            // Initialize convolution buffers (simplified)
            uint32_t max_ir_length = 65536; // 1.5 seconds at 44.1kHz
            convolution_buffers_.resize(channels_, std::vector<float>(max_ir_length, 0.0f));
            impulse_response_.resize(max_ir_length, 0.0f);
            ir_length_ = max_ir_length;
            break;
        }

        default:
            Logger::warn("Unsupported reverb algorithm: {}", static_cast<int>(current_algorithm_));
            break;
    }
}

void ReverbProcessor::process_freeverb(const float* input, float* output, uint32_t frame_count) {
    // Simplified Freeverb implementation
    // In a real implementation, would follow the exact Freeverb algorithm

    float room_size = parameters_.room_size;
    float damping = parameters_.damping;
    float width = parameters_.width;
    float wet_level = 0.5f;
    float dry_level = 0.5f;

    // Process through comb filters
    for (auto& delay_line : comb_delay_lines_) {
        if (delay_line) {
            delay_line->set_wet_mix(room_size);
            delay_line->set_feedback(1.0f - damping);
            delay_line->process_block(input, temp_buffer_.data(), frame_count, {});

            // Add to output
            for (uint32_t i = 0; i < frame_count * channels_; ++i) {
                output[i] += temp_buffer_[i] * wet_level;
            }
        }
    }

    // Process through all-pass filters
    for (auto& delay_line : allpass_delay_lines_) {
        if (delay_line) {
            delay_line->set_feedback(0.7f);
            delay_line->process_block(output, temp_buffer_.data(), frame_count, {});
            std::memcpy(output, temp_buffer_.data(), frame_count * channels_ * sizeof(float));
        }
    }
}

void ReverbProcessor::process_schroeder(const float* input, float* output, uint32_t frame_count) {
    // Schroeder reverb implementation placeholder
    // In a real implementation, would implement proper Schroeder reverberator
    process_freeverb(input, output, frame_count);
}

void ReverbProcessor::process_convolution(const float* input, float* output, uint32_t frame_count) {
    // Convolution reverb implementation placeholder
    // In a real implementation, would use FFT-based convolution for efficiency
    if (ir_length_ == 0 || impulse_response_.empty()) {
        std::memcpy(output, input, frame_count * channels_ * sizeof(float));
        return;
    }

    // Simple time-domain convolution (inefficient - for demo only)
    for (uint32_t ch = 0; ch < channels_; ++ch) {
        for (uint32_t i = 0; i < frame_count; ++i) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < ir_length_ && j <= i; ++j) {
                sum += input[ch * frame_count + i - j] * impulse_response_[j];
            }
            output[ch * frame_count + i] = sum;
        }
    }

    std::lock_guard<std::mutex> lock(stats_mutex_);
    statistics_.convolution_operations += frame_count * ir_length_;
}

void ReverbProcessor::update_algorithm_parameters() {
    switch (current_algorithm_) {
        case ReverbAlgorithm::FREEVERB:
            // Update comb filter feedback based on room size
            for (auto& delay_line : comb_delay_lines_) {
                if (delay_line) {
                    delay_line->set_feedback(parameters_.room_size * 0.98f);
                }
            }
            break;

        default:
            break;
    }
}

// ModulationProcessor implementation (simplified)
ModulationProcessor::ModulationProcessor()
    : modulation_type_(TimeDomainEffectType::CHORUS)
    , lfo_waveform_(ModulationWaveform::SINE)
    , bypassed_(false)
    , dry_wet_mix_(0.5f)
    , sample_rate_(44100)
    , channels_(2)
    , max_frame_size_(4096) {
}

ModulationProcessor::~ModulationProcessor() {
    shutdown();
}

bool ModulationProcessor::initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) {
    sample_rate_ = sample_rate;
    channels_ = channels;
    max_frame_size_ = max_frame_size;

    parameters_ = TimeDomainParameters();
    parameters_.modulation_rate_hz = 1.0f;
    parameters_.modulation_depth = 0.5f;
    parameters_.chorus_delay_ms = 20.0f;
    parameters_.stereo_spread_ms = 0.5f;

    initialize_lfos();
    initialize_delay_lines();

    uint32_t total_samples = max_frame_size * channels;
    wet_buffer_.resize(total_samples, 0.0f);
    dry_buffer_.resize(total_samples, 0.0f);
    mod_buffer_.resize(total_samples, 0.0f);

    return true;
}

void ModulationProcessor::shutdown() {
    delay_lines_.clear();
    lfos_.clear();
    wet_buffer_.clear();
    dry_buffer_.clear();
    mod_buffer_.clear();
}

bool ModulationProcessor::process(const float* input, float* output, uint32_t frame_count) {
    if (bypassed_) {
        std::memcpy(output, input, frame_count * channels_ * sizeof(float));
        return true;
    }

    std::memcpy(dry_buffer_.data(), input, frame_count * channels_ * sizeof(float));
    std::fill(wet_buffer_.begin(), wet_buffer_.begin() + frame_count * channels_, 0.0f);

    switch (modulation_type_) {
        case TimeDomainEffectType::CHORUS:
            process_chorus(input, wet_buffer_.data(), frame_count);
            break;
        case TimeDomainEffectType::FLANGER:
            process_flanger(input, wet_buffer_.data(), frame_count);
            break;
        case TimeDomainEffectType::PHASER:
            process_phaser(input, wet_buffer_.data(), frame_count);
            break;
        default:
            process_chorus(input, wet_buffer_.data(), frame_count);
            break;
    }

    float wet_gain = dry_wet_mix_;
    float dry_gain = 1.0f - wet_gain;
    for (uint32_t i = 0; i < frame_count * channels_; ++i) {
        output[i] = dry_buffer_[i] * dry_gain + wet_buffer_[i] * wet_gain;
    }

    return true;
}

void ModulationProcessor::initialize_lfos() {
    lfos_.resize(channels_);
    lfo_phase_offsets_.resize(channels_);

    for (uint32_t ch = 0; ch < channels_; ++ch) {
        lfos_[ch].phase = 0.0f;
        lfos_[ch].frequency = parameters_.modulation_rate_hz;
        lfos_[ch].depth = parameters_.modulation_depth;
        lfos_[ch].waveform = lfo_waveform_;
        lfo_phase_offsets_[ch] = static_cast<float>(ch) * 2.0f * M_PI / channels_; // Stereo spread
    }
}

void ModulationProcessor::initialize_delay_lines() {
    delay_lines_.resize(channels_);
    float max_delay_ms = parameters_.chorus_delay_ms * 2.0f; // Maximum delay for modulation
    uint32_t max_delay_samples = static_cast<uint32_t>(max_delay_ms * sample_rate_ / 1000.0f) + 1024;

    for (uint32_t ch = 0; ch < channels_; ++ch) {
        delay_lines_[ch] = std::make_unique<DelayLine>();
        delay_lines_[ch]->initialize(max_delay_samples, 1, InterpolationType::CUBIC);
        delay_lines_[ch]->set_wet_mix(1.0f);
    }
}

void ModulationProcessor::process_chorus(const float* input, float* output, uint32_t frame_count) {
    for (uint32_t frame = 0; frame < frame_count; ++frame) {
        for (uint32_t ch = 0; ch < channels_; ++ch) {
            uint32_t input_idx = frame * channels_ + ch;

            // Calculate LFO value
            float lfo_value = generate_lfo_sample(ch, parameters_.modulation_rate_hz / sample_rate_);

            // Calculate modulation delay
            float mod_delay = parameters_.chorus_delay_ms * (1.0f + lfo_value * parameters_.modulation_depth);
            float delay_samples = time_domain_utils::milliseconds_to_samples(mod_delay, sample_rate_);

            // Read from delay line
            float delayed_sample = delay_lines_[ch]->read_sample(0, delay_samples);

            // Write to delay line
            delay_lines_[ch]->write_sample(0, input[input_idx]);

            // Mix dry and wet
            output[input_idx] = delayed_sample;
        }
    }
}

float ModulationProcessor::generate_lfo_sample(uint32_t lfo_index, float phase_increment) {
    auto& lfo = lfos_[lfo_index];
    lfo.phase += phase_increment + lfo_phase_offsets_[lfo_index];

    if (lfo.phase >= 1.0f) {
        lfo.phase -= 1.0f;
    }

    return get_lfo_value(lfo.waveform, lfo.phase * 2.0f * M_PI);
}

float ModulationProcessor::get_lfo_value(ModulationWaveform waveform, float phase) {
    switch (waveform) {
        case ModulationWaveform::SINE:
            return std::sin(phase);
        case ModulationWaveform::TRIANGLE:
            return 2.0f * std::abs(2.0f * (phase / (2.0f * M_PI) - std::floor(phase / (2.0f * M_PI) + 0.5f))) - 1.0f;
        case ModulationWaveform::SAWTOOTH:
            return 2.0f * (phase / (2.0f * M_PI) - std::floor(phase / (2.0f * M_PI) + 0.5f));
        case ModulationWaveform::SQUARE:
            return (std::sin(phase) >= 0.0f) ? 1.0f : -1.0f;
        default:
            return std::sin(phase);
    }
}

// Factory implementations
std::unique_ptr<TimeDomainEffect> TimeDomainEffectsFactory::create_simple_delay(float max_delay_ms) {
    auto processor = std::make_unique<MultiTapDelayProcessor>();
    TimeDomainParameters params;
    params.num_taps = 1;
    params.tap_delay_times_ms = {max_delay_ms / 2.0f};
    params.tap_gain_levels = {1.0f};
    params.tap_feedback_levels = {0.3f};

    processor->set_parameters(params);
    return std::move(processor);
}

std::unique_ptr<TimeDomainEffect> TimeDomainEffectsFactory::create_multi_tap_delay(uint32_t num_taps) {
    auto processor = std::make_unique<MultiTapDelayProcessor>();
    processor->set_num_taps(num_taps);
    return std::move(processor);
}

std::unique_ptr<TimeDomainEffect> TimeDomainEffectsFactory::create_chorus() {
    auto processor = std::make_unique<ModulationProcessor>();
    processor->set_modulation_type(TimeDomainEffectType::CHORUS);
    return std::move(processor);
}

std::unique_ptr<TimeDomainEffect> TimeDomainEffectsFactory::create_flanger() {
    auto processor = std::make_unique<ModulationProcessor>();
    processor->set_modulation_type(TimeDomainEffectType::FLANGER);
    return std::move(processor);
}

std::unique_ptr<TimeDomainEffect> TimeDomainEffectsFactory::create_phaser() {
    auto processor = std::make_unique<ModulationProcessor>();
    processor->set_modulation_type(TimeDomainEffectType::PHASER);
    return std::move(processor);
}

std::unique_ptr<TimeDomainEffect> TimeDomainEffectsFactory::create_hall_reverb() {
    auto processor = std::make_unique<ReverbProcessor>();
    TimeDomainParameters params;
    params.reverb_algorithm = ReverbAlgorithm::FREEVERB;
    params.room_size = 0.8f;
    params.damping = 0.3f;
    params.width = 1.0f;

    processor->set_parameters(params);
    return std::move(processor);
}

std::unique_ptr<TimeDomainEffect> TimeDomainEffectsFactory::create_room_reverb() {
    auto processor = std::make_unique<ReverbProcessor>();
    TimeDomainParameters params;
    params.reverb_algorithm = ReverbAlgorithm::FREEVERB;
    params.room_size = 0.4f;
    params.damping = 0.7f;
    params.width = 0.8f;

    processor->set_parameters(params);
    return std::move(processor);
}

std::vector<std::string> TimeDomainEffectsFactory::get_available_effect_types() {
    return {
        "delay", "multi_tap_delay", "ping_pong_delay", "slapback_delay",
        "reverb", "hall_reverb", "room_reverb", "plate_reverb", "spring_reverb",
        "chorus", "flanger", "phaser", "vibrato", "tremolo",
        "pitch_shifter", "time_stretch", "doubler"
    };
}

// Utility namespace implementations
namespace time_domain_utils {

float milliseconds_to_samples(float time_ms, uint32_t sample_rate) {
    return time_ms * sample_rate / 1000.0f;
}

float samples_to_milliseconds(uint32_t samples, uint32_t sample_rate) {
    return static_cast<float>(samples) * 1000.0f / sample_rate;
}

float linear_interpolate(float a, float b, float fraction) {
    return a + (b - a) * fraction;
}

float cubic_interpolate(float a, float b, float c, float d, float fraction) {
    float fraction_sq = fraction * fraction;
    float fraction_cu = fraction_sq * fraction;

    return b + 0.5f * fraction * (c - a) +
           fraction_sq * (a - 2.5f * b + 2.0f * c - 0.5f * d) +
           fraction_cu * (0.5f * (d - a) + 1.5f * (b - c));
}

float hermite_interpolate(float a, float b, float c, float d, float fraction) {
    float fraction_sq = fraction * fraction;
    float fraction_cu = fraction_sq * fraction;

    float tension = 0.0f; // Tension parameter (-1 to 1)
    float bias = 0.0f;    // Bias parameter (-1 to 1)

    float m0 = (b - a) * (1.0f + bias) * (1.0f - tension) / 2.0f +
               (c - b) * (1.0f - bias) * (1.0f - tension) / 2.0f;
    float m1 = (c - b) * (1.0f + bias) * (1.0f - tension) / 2.0f +
               (d - c) * (1.0f - bias) * (1.0f - tension) / 2.0f;

    return a + (b - a) * fraction +
           (m0 * (fraction - fraction_sq) + (m1 * fraction_sq)) +
           ((b - a + m0) * fraction_cu - (2.0f * b - 2.0f * c + m0 + m1) * fraction_cu);
}

void calculate_low_pass_coefficients(float frequency, float sample_rate, float resonance,
                                     float& a0, float& a1, float& a2, float& b1, float& b2) {
    float w = 2.0f * M_PI * frequency / sample_rate;
    float q = std::sqrt(resonance);
    float alpha = std::sin(w) / (2.0f * q);
    float cosw = std::cos(w);

    a0 = 1.0f + alpha;
    a1 = -2.0f * cosw;
    a2 = 1.0f - alpha;
    b1 = -2.0f * cosw;
    b2 = 1.0f - alpha;

    // Normalize
    a0 = 1.0f / a0;
    a1 *= a0;
    a2 *= a0;
    b1 *= a0;
    b2 *= a0;
}

void calculate_freeverb_comb_tuning(float sample_rate, std::vector<float>& comb_tuning) {
    comb_tuning = {
        (1117.0f / sample_rate), // comb1
        (1188.0f / sample_rate), // comb2
        (1277.0f / sample_rate), // comb3
        (1356.0f / sample_rate), // comb4
        (1422.0f / sample_rate), // comb5
        (1491.0f / sample_rate), // comb6
        (1557.0f / sample_rate), // comb7
        (1617.0f / sample_rate)  // comb8
    };
}

void calculate_freeverb_allpass_tuning(float sample_rate, std::vector<float>& allpass_tuning) {
    allpass_tuning = {
        (556.0f / sample_rate),  // allpass1
        (441.0f / sample_rate),  // allpass2
        (341.0f / sample_rate),  // allpass3
        (225.0f / sample_rate)   // allpass4
    };
}

bool is_sse_supported() {
#ifdef __SSE__
    return true;
#else
    return false;
#endif
}

bool is_avx_supported() {
#ifdef __AVX__
    return true;
#else
    return false;
#endif
}

void* aligned_malloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

} // namespace time_domain_utils

} // namespace vortex::core::dsp