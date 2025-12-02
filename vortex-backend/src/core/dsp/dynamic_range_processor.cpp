#include "core/dsp/dynamic_range_processor.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <random>
#include <immintrin.h>

namespace vortex::core::dsp {

// Compressor implementation
Compressor::Compressor()
    : type_(DynamicRangeType::COMPRESSOR)
    , model_(CompressionModel::DIGITAL)
    , bypassed_(false)
    , enabled_(true)
    , sample_rate_(44100)
    , channels_(2)
    , max_frame_size_(4096)
    , current_gain_reduction_db_(0.0f)
    , envelope_value_(0.0f)
    , input_level_dbfs_(0.0f)
    , output_level_dbfs_(0.0f)
    , attack_coefficient_(0.0f)
    , release_coefficient_(0.0f)
    , smoothing_coefficient_(0.999f)
    , look_ahead_samples_(0)
    , look_ahead_index_(0)
    , sidechain_level_dbfs_(0.0f) {
}

Compressor::~Compressor() {
    shutdown();
}

bool Compressor::initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    sample_rate_ = sample_rate;
    channels_ = channels;
    max_frame_size_ = max_frame_size;

    Logger::info("Initializing Compressor: {} Hz, {} channels", sample_rate, channels);

    // Initialize parameters
    params_ = DynamicRangeParameters();
    params_.type = DynamicRangeType::COMPRESSOR;
    params_.threshold_dbfs = -20.0f;
    params_.ratio = 4.0f;
    params_.attack_time_ms = 5.0f;
    params_.release_time_ms = 100.0f;
    params_.knee_width_db = 2.0f;
    params_.makeup_gain_db = 0.0f;

    // Initialize channel states
    channel_gain_reductions_.resize(channels_, 0.0f);
    channel_envelopes_.resize(channels_, 0.0f);

    // Calculate initial coefficients
    calculate_attack_release_coefficients();

    // Initialize look-ahead buffer
    if (params_.look_ahead_time_ms > 0.0f) {
        look_ahead_samples_ = static_cast<uint32_t>(params_.look_ahead_time_ms * sample_rate / 1000.0f);
        look_ahead_buffer_.resize(look_ahead_samples_ * channels_, 0.0f);
        look_ahead_delay_buffer_.resize(look_ahead_samples_ * channels_, 0.0f);
        look_ahead_index_ = 0;
    }

    // Initialize sidechain buffer
    sidechain_buffer_.resize(max_frame_size * channels_, 0.0f);
    sidechain_filter_state_.resize(channels_ * 4, 0.0f); // 4th order filter

    // Initialize processing buffers
    uint32_t total_samples = max_frame_size * channels_;
    input_buffer_.resize(total_samples, 0.0f);
    output_buffer_.resize(total_samples, 0.0f);
    gain_buffer_.resize(max_frame_size, 0.0f);

    // Reset statistics
    statistics_ = DynamicRangeStatistics();
    statistics_.last_reset_time = std::chrono::steady_clock::now();

    // Initialize presets
    presets_["vocal"] = params_;
    presets_["vocal"].threshold_dbfs = -18.0f;
    presets_["vocal"].ratio = 3.0f;
    presets_["vocal"].attack_time_ms = 3.0f;
    presets_["vocal"].release_time_ms = 150.0f;

    presets_["mastering"] = params_;
    presets_["mastering"].threshold_dbfs = -8.0f;
    presets_["mastering"].ratio = 2.0f;
    presets_["mastering"].attack_time_ms = 10.0f;
    presets_["mastering"].release_time_ms = 500.0f;
    presets_["mastering"].knee_width_db = 4.0f;

    presets_["drums"] = params_;
    presets_["drums"].threshold_dbfs = -12.0f;
    presets_["drums"].ratio = 6.0f;
    presets_["drums"].attack_time_ms = 1.0f;
    presets_["drums"].release_time_ms = 50.0f;

    Logger::info("Compressor initialized successfully");
    return true;
}

void Compressor::shutdown() {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    look_ahead_buffer_.clear();
    look_ahead_delay_buffer_.clear();
    sidechain_buffer_.clear();
    sidechain_filter_state_.clear();
    input_buffer_.clear();
    output_buffer_.clear();
    gain_buffer_.clear();
    channel_gain_reductions_.clear();
    channel_envelopes_.clear();
    presets_.clear();

    Logger::info("Compressor shutdown");
}

bool Compressor::reset() {
    current_gain_reduction_db_ = 0.0f;
    envelope_value_ = 0.0f;
    input_level_dbfs_ = 0.0f;
    output_level_dbfs_ = 0.0f;

    std::fill(channel_gain_reductions_.begin(), channel_gain_reductions_.end(), 0.0f);
    std::fill(channel_envelopes_.begin(), channel_envelopes_.end(), 0.0f);

    std::fill(look_ahead_buffer_.begin(), look_ahead_buffer_.end(), 0.0f);
    std::fill(look_ahead_delay_buffer_.begin(), look_ahead_delay_buffer_.end(), 0.0f);
    std::fill(sidechain_filter_state_.begin(), sidechain_filter_state_.end(), 0.0f);
    std::fill(gain_buffer_.begin(), gain_buffer_.end(), 0.0f);

    look_ahead_index_ = 0;

    Logger::info("Compressor reset");
    return true;
}

bool Compressor::process(const float* input, float* output, uint32_t frame_count) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (!input || !output || frame_count == 0 || bypassed_) {
        if (input && output && frame_count > 0) {
            std::memcpy(output, input, frame_count * channels_ * sizeof(float));
        }
        return true;
    }

    // Store input buffer for analysis
    std::memcpy(input_buffer_.data(), input, frame_count * channels_ * sizeof(float));

    // Calculate input level
    float max_input_sample = 0.0f;
    for (uint32_t i = 0; i < frame_count * channels_; ++i) {
        max_input_sample = std::max(max_input_sample, std::abs(input[i]));
    }
    input_level_dbfs_ = dynamic_range_utils::linear_to_dbfs(max_input_sample);

    // Process each sample
    for (uint32_t frame = 0; frame < frame_count; ++frame) {
        float max_channel_level = 0.0f;
        float max_channel_gain = 0.0f;

        // Process each channel
        for (uint32_t ch = 0; ch < channels_; ++ch) {
            uint32_t sample_index = frame * channels_ + ch;
            float input_sample = input[sample_index];

            // Sidechain processing if enabled
            if (params_.enable_sidechain) {
                float sidechain_sample = process_sidechain_filter(input_sample, ch);
                max_channel_level = std::max(max_channel_level, sidechain_sample);
            } else {
                max_channel_level = std::max(max_channel_level, std::abs(input_sample));
            }
        }

        // Convert to dBFS
        float level_db = dynamic_range_utils::linear_to_dbfs(max_channel_level);
        envelope_value_ = smooth_parameter(envelope_value_, level_db, smoothing_coefficient_);

        // Calculate gain reduction
        float gain_reduction = calculate_gain_reduction(envelope_value_, params_.threshold_dbfs,
                                                        params_.ratio, params_.knee_width_db);

        // Apply attack/release smoothing
        float target_gain_reduction = gain_reduction;
        if (target_gain_reduction > current_gain_reduction_db_) {
            // Attack
            current_gain_reduction_db_ = current_gain_reduction_db_ +
                (target_gain_reduction - current_gain_reduction_db_) * attack_coefficient_;
        } else {
            // Release
            current_gain_reduction_db_ = current_gain_reduction_db_ +
                (target_gain_reduction - current_gain_reduction_db_) * release_coefficient_;
        }

        // Convert gain reduction to linear gain
        float gain_linear = dynamic_range_utils::dbfs_to_linear(-current_gain_reduction_db_);

        // Apply makeup gain
        gain_linear *= dynamic_range_utils::dbfs_to_linear(params_.makeup_gain_db);
        max_channel_gain = gain_linear;

        // Apply gain to each channel
        for (uint32_t ch = 0; ch < channels_; ++ch) {
            uint32_t sample_index = frame * channels_ + ch;
            float processed_sample = input[sample_index] * gain_linear;

            // Apply limiting if enabled
            if (params_.enable_limiting && processed_sample > 1.0f) {
                processed_sample = 1.0f;
            } else if (params_.enable_limiting && processed_sample < -1.0f) {
                processed_sample = -1.0f;
            }

            output[sample_index] = processed_sample;
        }

        gain_buffer_[frame] = -current_gain_reduction_db_;
    }

    // Calculate output level
    float max_output_sample = 0.0f;
    for (uint32_t i = 0; i < frame_count * channels_; ++i) {
        max_output_sample = std::max(max_output_sample, std::abs(output[i]));
    }
    output_level_dbfs_ = dynamic_range_utils::linear_to_dbfs(max_output_sample);

    // Update statistics
    update_statistics(input, output, frame_count);

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

bool Compressor::process_interleaved(const float* input, float* output, uint32_t frame_count) {
    // For this implementation, interleaved and non-interleaved processing are the same
    return process(input, output, frame_count);
}

bool Compressor::process_with_sidechain(const float* input, const float* sidechain,
                                       float* output, uint32_t frame_count) {
    if (!sidechain || !params_.enable_sidechain) {
        return process(input, output, frame_count);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    if (!input || !output || !sidechain || frame_count == 0 || bypassed_) {
        if (input && output && frame_count > 0) {
            std::memcpy(output, input, frame_count * channels_ * sizeof(float));
        }
        return true;
    }

    // Store input buffer for analysis
    std::memcpy(input_buffer_.data(), input, frame_count * channels_ * sizeof(float));

    // Process each sample with sidechain
    for (uint32_t frame = 0; frame < frame_count; ++frame) {
        float max_sidechain_level = 0.0f;

        // Calculate sidechain level from sidechain input
        for (uint32_t ch = 0; ch < channels_; ++ch) {
            uint32_t sc_index = frame * channels_ + ch;
            float sidechain_sample = process_sidechain_filter(sidechain[sc_index], ch);
            max_sidechain_level = std::max(max_sidechain_level, std::abs(sidechain_sample));
        }

        // Convert to dBFS
        float level_db = dynamic_range_utils::linear_to_dbfs(max_sidechain_level);
        envelope_value_ = smooth_parameter(envelope_value_, level_db, smoothing_coefficient_);

        // Calculate gain reduction based on sidechain
        float gain_reduction = calculate_gain_reduction(envelope_value_, params_.threshold_dbfs,
                                                        params_.ratio, params_.knee_width_db);

        // Apply attack/release smoothing
        float target_gain_reduction = gain_reduction;
        if (target_gain_reduction > current_gain_reduction_db_) {
            current_gain_reduction_db_ = current_gain_reduction_db_ +
                (target_gain_reduction - current_gain_reduction_db_) * attack_coefficient_;
        } else {
            current_gain_reduction_db_ = current_gain_reduction_db_ +
                (target_gain_reduction - current_gain_reduction_db_) * release_coefficient_;
        }

        // Convert gain reduction to linear gain
        float gain_linear = dynamic_range_utils::dbfs_to_linear(-current_gain_reduction_db_);

        // Apply makeup gain and ducking
        if (params_.enable_ducking) {
            float ducking_gain = dynamic_range_utils::dbfs_to_linear(-params_.ducking_depth_db);
            gain_linear *= ducking_gain;
        } else {
            gain_linear *= dynamic_range_utils::dbfs_to_linear(params_.makeup_gain_db);
        }

        // Apply gain to input signal
        for (uint32_t ch = 0; ch < channels_; ++ch) {
            uint32_t sample_index = frame * channels_ + ch;
            float processed_sample = input[sample_index] * gain_linear;

            // Apply limiting if enabled
            if (params_.enable_limiting && processed_sample > 1.0f) {
                processed_sample = 1.0f;
            } else if (params_.enable_limiting && processed_sample < -1.0f) {
                processed_sample = -1.0f;
            }

            output[sample_index] = processed_sample;
        }
    }

    // Update statistics
    update_statistics(input, output, frame_count);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::lock_guard<std::mutex> lock(stats_mutex_);
    statistics_.total_process_calls++;
    statistics_.successful_calls++;

    double processing_time_us = duration.count();
    statistics_.avg_processing_time_us =
        ((statistics_.avg_processing_time_us * (statistics_.total_process_calls - 1)) + processing_time_us) /
        statistics_.total_process_calls;

    return true;
}

bool Compressor::set_parameters(const DynamicRangeParameters& params) {
    params_ = params;
    type_ = params.type;
    model_ = params.model;

    // Recalculate coefficients
    calculate_attack_release_coefficients();

    // Update look-ahead buffer size
    if (params_.look_ahead_time_ms > 0.0f) {
        uint32_t new_lookahead_samples = static_cast<uint32_t>(params_.look_ahead_time_ms * sample_rate_ / 1000.0f);
        if (new_lookahead_samples != look_ahead_samples_) {
            look_ahead_samples_ = new_lookahead_samples;
            look_ahead_buffer_.resize(look_ahead_samples_ * channels_, 0.0f);
            look_ahead_delay_buffer_.resize(look_ahead_samples_ * channels_, 0.0f);
            look_ahead_index_ = 0;
        }
    }

    Logger::info("Updated compressor parameters: threshold={:.1f}dB, ratio={:.1f}:1",
                params_.threshold_dbfs, params_.ratio);
    return true;
}

DynamicRangeParameters Compressor::get_parameters() const {
    return params_;
}

bool Compressor::set_parameter(const std::string& name, float value) {
    bool coeff_changed = false;

    if (name == "threshold_dbfs") {
        params_.threshold_dbfs = value;
    } else if (name == "ratio") {
        params_.ratio = std::max(1.0f, value);
    } else if (name == "attack_time_ms") {
        params_.attack_time_ms = std::max(0.1f, value);
        coeff_changed = true;
    } else if (name == "release_time_ms") {
        params_.release_time_ms = std::max(0.1f, value);
        coeff_changed = true;
    } else if (name == "knee_width_db") {
        params_.knee_width_db = std::max(0.0f, value);
    } else if (name == "makeup_gain_db") {
        params_.makeup_gain_db = value;
    } else if (name == "enabled") {
        enabled_ = (value != 0.0f);
    } else if (name == "bypassed") {
        bypassed_ = (value != 0.0f);
    } else {
        return false;
    }

    if (coeff_changed) {
        calculate_attack_release_coefficients();
    }

    return true;
}

float Compressor::get_parameter(const std::string& name) const {
    if (name == "threshold_dbfs") {
        return params_.threshold_dbfs;
    } else if (name == "ratio") {
        return params_.ratio;
    } else if (name == "attack_time_ms") {
        return params_.attack_time_ms;
    } else if (name == "release_time_ms") {
        return params_.release_time_ms;
    } else if (name == "knee_width_db") {
        return params_.knee_width_db;
    } else if (name == "makeup_gain_db") {
        return params_.makeup_gain_db;
    } else if (name == "enabled") {
        return enabled_ ? 1.0f : 0.0f;
    } else if (name == "bypassed") {
        return bypassed_ ? 1.0f : 0.0f;
    }
    return 0.0f;
}

bool Compressor::set_threshold(float threshold_dbfs) {
    params_.threshold_dbfs = threshold_dbfs;
    return true;
}

bool Compressor::set_ratio(float ratio) {
    params_.ratio = std::max(1.0f, ratio);
    return true;
}

bool Compressor::set_attack_time(float attack_ms) {
    params_.attack_time_ms = std::max(0.1f, attack_ms);
    calculate_attack_release_coefficients();
    return true;
}

bool Compressor::set_release_time(float release_ms) {
    params_.release_time_ms = std::max(0.1f, release_ms);
    calculate_attack_release_coefficients();
    return true;
}

bool Compressor::set_makeup_gain(float gain_db) {
    params_.makeup_gain_db = gain_db;
    return true;
}

bool Compressor::set_knee_width(float width_db) {
    params_.knee_width_db = std::max(0.0f, width_db);
    return true;
}

bool Compressor::set_bypass(bool bypass) {
    bypassed_ = bypass;
    Logger::info("Compressor bypass {}", bypass ? "enabled" : "disabled");
    return true;
}

bool Compressor::is_bypassed() const {
    return bypassed_;
}

bool Compressor::set_enabled(bool enabled) {
    enabled_ = enabled;
    return true;
}

bool Compressor::is_enabled() const {
    return enabled_;
}

bool Compressor::save_preset(const std::string& name) {
    presets_[name] = params_;
    Logger::info("Saved preset: {}", name);
    return true;
}

bool Compressor::load_preset(const std::string& name) {
    auto it = presets_.find(name);
    if (it != presets_.end()) {
        params_ = it->second;
        calculate_attack_release_coefficients();
        Logger::info("Loaded preset: {}", name);
        return true;
    }
    Logger::warn("Preset not found: {}", name);
    return false;
}

std::vector<std::string> Compressor::get_available_presets() const {
    std::vector<std::string> preset_names;
    for (const auto& preset : presets_) {
        preset_names.push_back(preset.first);
    }
    return preset_names;
}

DynamicRangeType Compressor::get_type() const {
    return type_;
}

CompressionModel Compressor::get_model() const {
    return model_;
}

std::string Compressor::get_name() const {
    return "Compressor";
}

std::string Compressor::get_version() const {
    return "1.0.0";
}

std::string Compressor::get_description() const {
    return "Professional dynamic range compressor with advanced features";
}

DynamicRangeStatistics Compressor::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return statistics_;
}

void Compressor::reset_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    statistics_ = DynamicRangeStatistics();
    statistics_.last_reset_time = std::chrono::steady_clock::now();
}

bool Compressor::supports_real_time_parameter_changes() const {
    return true;
}

bool Compressor::supports_gpu_acceleration() const {
    return false; // Dynamic range processing typically doesn't benefit much from GPU
}

bool Compressor::is_multiband() const {
    return false;
}

bool Compressor::has_sidechain() const {
    return true;
}

double Compressor::get_expected_latency_ms() const {
    return (static_cast<double>(params_.look_ahead_time_ms) +
            (params_.attack_time_ms / 10.0)); // Small additional latency for processing
}

float Compressor::get_current_gain_reduction() const {
    return current_gain_reduction_db_;
}

float Compressor::get_input_level() const {
    return input_level_dbfs_;
}

float Compressor::get_output_level() const {
    return output_level_dbfs_;
}

float Compressor::get_envelope_follower_value() const {
    return envelope_value_;
}

bool Compressor::is_clipping() const {
    return output_level_dbfs_ >= 0.0f;
}

void Compressor::calculate_attack_release_coefficients() {
    attack_coefficient_ = dynamic_range_utils::attack_time_to_coefficient(
        params_.attack_time_ms, sample_rate_);
    release_coefficient_ = dynamic_range_utils::release_time_to_coefficient(
        params_.release_time_ms, sample_rate_);
}

float Compressor::calculate_gain_reduction(float input_level_db, float threshold_db,
                                           float ratio, float knee_width) {
    if (input_level_db <= threshold_db - knee_width / 2.0f) {
        return 0.0f; // No compression
    }

    float over_threshold = input_level_db - threshold_db;

    if (params_.knee_type == KneeType::HARD_KNEE) {
        // Hard knee compression
        if (ratio > 1.0f) {
            return over_threshold * (1.0f - 1.0f / ratio);
        }
    } else if (params_.knee_type == KneeType::SOFT_KNEE) {
        // Soft knee compression
        float knee_start = threshold_db - knee_width / 2.0f;
        float knee_end = threshold_db + knee_width / 2.0f;

        if (input_level_db < knee_start) {
            return 0.0f;
        } else if (input_level_db > knee_end) {
            over_threshold = input_level_db - threshold_db;
            return over_threshold * (1.0f - 1.0f / ratio);
        } else {
            // Soft knee region - linear interpolation
            float knee_ratio = 1.0f + (ratio - 1.0f) *
                             (input_level_db - knee_start) / knee_width;
            float gain_reduction = over_threshold * (1.0f - 1.0f / knee_ratio);
            return gain_reduction;
        }
    }

    return 0.0f;
}

float Compressor::process_sidechain_filter(float input, uint32_t channel) {
    if (!params_.enable_sidechain || params_.sidechain_filter_type == SidechainFilterType::NONE) {
        return input;
    }

    // Simple one-pole low-pass filter implementation
    // In a real implementation, would use proper filter design based on filter_type
    float cutoff_normalized = params_.sidechain_freq_hz / (sample_rate_ / 2.0f);
    float alpha = std::min(0.99f, cutoff_normalized / (1.0f + cutoff_normalized));

    uint32_t state_index = channel * 4; // 4 filter states per channel
    float& state1 = sidechain_filter_state_[state_index];
    float& state2 = sidechain_filter_state_[state_index + 1];

    if (params_.sidechain_filter_type == SidechainFilterType::LOW_PASS) {
        state1 = state1 + alpha * (input - state1);
        return state1;
    } else if (params_.sidechain_filter_type == SidechainFilterType::HIGH_PASS) {
        state2 = alpha * (state2 + input - state1);
        state1 = alpha * (state1 + input);
        return input - state1;
    }

    return input;
}

void Compressor::update_statistics(const float* input, const float* output, uint32_t frame_count) {
    // Calculate RMS levels
    float input_rms = 0.0f;
    float output_rms = 0.0f;

    for (uint32_t i = 0; i < frame_count * channels_; ++i) {
        input_rms += input[i] * input[i];
        output_rms += output[i] * output[i];
    }

    uint32_t total_samples = frame_count * channels_;
    input_rms = std::sqrt(input_rms / total_samples);
    output_rms = std::sqrt(output_rms / total_samples);

    float input_rms_dbfs = dynamic_range_utils::linear_to_dbfs(input_rms);
    float output_rms_dbfs = dynamic_range_utils::linear_to_dbfs(output_rms);

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        statistics_.input_level_dbfs = input_rms_dbfs;
        statistics_.output_level_dbfs = output_rms_dbfs;
        statistics_.current_gain_reduction_db = current_gain_reduction_db_;
        statistics_.max_gain_reduction_db = std::max(statistics_.max_gain_reduction_db,
                                                   current_gain_reduction_db_);
        statistics_.avg_gain_reduction_db =
            (statistics_.avg_gain_reduction_db * (statistics_.total_process_calls - 1) +
             current_gain_reduction_db) / statistics_.total_process_calls;
        statistics_.makeup_gain_db = params_.makeup_gain_db;
        statistics_.envelope_follower_value = envelope_value_;
        statistics_.sidechain_level_dbfs = sidechain_level_dbfs;
        statistics_.attack_rate_db_per_sec = 60.0f / params_.attack_time_ms;
        statistics_.release_rate_db_per_sec = 60.0f / params_.release_time_ms;
        statistics_.look_ahead_samples = look_ahead_samples_;
        statistics_.dynamic_range_db = input_rms_dbfs - output_rms_dbfs;
        statistics_.is_compressing = (current_gain_reduction_db > 0.1f);
        statistics_.is_limiting = (output_level_dbfs >= params_.ceiling_dbfs - 1.0f);
    }

    // Check for clipping
    if (check_clipping(output, frame_count)) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        statistics_.clipping_events++;
    }
}

float Compressor::smooth_parameter(float current, float target, float coefficient) {
    return current + (target - current) * (1.0f - coefficient);
}

bool Compressor::check_clipping(const float* buffer, uint32_t frame_count) {
    for (uint32_t i = 0; i < frame_count * channels_; ++i) {
        if (std::abs(buffer[i]) > 1.0f) {
            return true;
        }
    }
    return false;
}

// Limiter implementation (simplified)
Limiter::Limiter()
    : type_(DynamicRangeType::LIMITER)
    , model_(CompressionModel::DIGITAL)
    , bypassed_(false)
    , enabled_(true)
    , sample_rate_(44100)
    , channels_(2)
    , max_frame_size_(4096)
    , ceiling_dbfs_(-0.1f)
    , current_gain_db_(0.0f)
    , envelope_value_(0.0f)
    , brickwall_mode_(false)
    , attack_coefficient_(0.0f)
    , release_coefficient_(0.0f)
    , look_ahead_samples_(0)
    , look_ahead_index_(0) {
}

bool Limiter::initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) {
    sample_rate_ = sample_rate;
    channels_ = channels;
    max_frame_size_ = max_frame_size;

    params_ = DynamicRangeParameters();
    params_.type = DynamicRangeType::LIMITER;
    params_.threshold_dbfs = -3.0f;
    params_.ratio = 100.0f; // Very high ratio for limiting
    params_.attack_time_ms = 1.0f;
    params_.release_time_ms = 10.0f;
    params_.makeup_gain_db = 0.0f;

    calculate_attack_release_coefficients();

    uint32_t total_samples = max_frame_size * channels_;
    input_buffer_.resize(total_samples, 0.0f);
    output_buffer_.resize(total_samples, 0.0f);

    return true;
}

void Limiter::shutdown() {
    input_buffer_.clear();
    output_buffer_.clear();
}

bool Limiter::reset() {
    current_gain_db_ = 0.0f;
    envelope_value_ = 0.0f;
    return true;
}

bool Limiter::process(const float* input, float* output, uint32_t frame_count) {
    if (bypassed_ || !input || !output || frame_count == 0) {
        if (input && output && frame_count > 0) {
            std::memcpy(output, input, frame_count * channels_ * sizeof(float));
        }
        return true;
    }

    // Simple limiting implementation
    for (uint32_t i = 0; i < frame_count * channels_; ++i) {
        float input_sample = input[i];
        float ceiling_linear = dynamic_range_utils::dbfs_to_linear(ceiling_dbfs_);

        if (std::abs(input_sample) > ceiling_linear) {
            output[i] = (input_sample > 0.0f) ? ceiling_linear : -ceiling_linear;
        } else {
            output[i] = input_sample;
        }
    }

    return true;
}

bool Limiter::process_interleaved(const float* input, float* output, uint32_t frame_count) {
    return process(input, output, frame_count);
}

bool Limiter::process_with_sidechain(const float* input, const float* sidechain,
                                     float* output, uint32_t frame_count) {
    return process(input, output, frame_count);
}

bool Limiter::set_parameters(const DynamicRangeParameters& params) {
    params_ = params;
    calculate_attack_release_coefficients();
    return true;
}

DynamicRangeParameters Limiter::get_parameters() const {
    return params_;
}

bool Limiter::set_parameter(const std::string& name, float value) {
    if (name == "threshold_dbfs") {
        params_.threshold_dbfs = value;
    } else if (name == "attack_time_ms") {
        params_.attack_time_ms = value;
        calculate_attack_release_coefficients();
    } else if (name == "release_time_ms") {
        params_.release_time_ms = value;
        calculate_attack_release_coefficients();
    } else {
        return false;
    }
    return true;
}

float Limiter::get_parameter(const std::string& name) const {
    if (name == "threshold_dbfs") {
        return params_.threshold_dbfs;
    } else if (name == "attack_time_ms") {
        return params_.attack_time_ms;
    } else if (name == "release_time_ms") {
        return params_.release_time_ms;
    }
    return 0.0f;
}

bool Limiter::set_threshold(float threshold_dbfs) {
    params_.threshold_dbfs = threshold_dbfs;
    return true;
}

bool Limiter::set_ratio(float ratio) {
    params_.ratio = std::max(10.0f, ratio);
    return true;
}

bool Limiter::set_attack_time(float attack_ms) {
    params_.attack_time_ms = attack_ms;
    calculate_attack_release_coefficients();
    return true;
}

bool Limiter::set_release_time(float release_ms) {
    params_.release_time_ms = release_ms;
    calculate_attack_release_coefficients();
    return true;
}

bool Limiter::set_makeup_gain(float gain_db) {
    params_.makeup_gain_db = gain_db;
    return true;
}

bool Limiter::set_knee_width(float width_db) {
    // Limiters typically use hard knee
    return false;
}

bool Limiter::set_bypass(bool bypass) {
    bypassed_ = bypass;
    return true;
}

bool Limiter::is_bypassed() const {
    return bypassed_;
}

bool Limiter::set_enabled(bool enabled) {
    enabled_ = enabled;
    return true;
}

bool Limiter::is_enabled() const {
    return enabled_;
}

bool Limiter::save_preset(const std::string& name) {
    // Preset implementation would go here
    return true;
}

bool Limiter::load_preset(const std::string& name) {
    // Preset implementation would go here
    return true;
}

std::vector<std::string> Limiter::get_available_presets() const {
    return {"default", "brickwall", "smooth"};
}

DynamicRangeType Limiter::get_type() const {
    return type_;
}

CompressionModel Limiter::get_model() const {
    return model_;
}

std::string Limiter::get_name() const {
    return "Limiter";
}

std::string Limiter::get_version() const {
    return "1.0.0";
}

std::string Limiter::get_description() const {
    return "Professional peak limiter with brickwall and soft limiting modes";
}

DynamicRangeStatistics Limiter::get_statistics() const {
    return DynamicRangeStatistics(); // Placeholder
}

void Limiter::reset_statistics() {
    // Statistics reset would go here
}

bool Limiter::supports_real_time_parameter_changes() const {
    return true;
}

bool Limiter::supports_gpu_acceleration() const {
    return false;
}

bool Limiter::is_multiband() const {
    return false;
}

bool Limiter::has_sidechain() const {
    return false;
}

double Limiter::get_expected_latency_ms() const {
    return params_.attack_time_ms / 10.0;
}

float Limiter::get_current_gain_reduction() const {
    return current_gain_db_;
}

float Limiter::get_input_level() const {
    return 0.0f; // Placeholder
}

float Limiter::get_output_level() const {
    return 0.0f; // Placeholder
}

float Limiter::get_envelope_follower_value() const {
    return envelope_value_;
}

bool Limiter::is_clipping() const {
    return false;
}

void Limiter::set_ceiling(float ceiling_dbfs) {
    ceiling_dbfs_ = ceiling_dbfs;
}

void Limiter::set_release_time_ms(float release_ms) {
    params_.release_time_ms = release_ms;
    calculate_attack_release_coefficients();
}

void Limiter::set_brickwall_mode(bool enabled) {
    brickwall_mode_ = enabled;
}

bool Limiter::is_brickwall_mode() const {
    return brickwall_mode_;
}

float Limiter::calculate_limiter_gain(float input_level_db, float ceiling_db, bool brickwall) {
    if (brickwall) {
        return std::min(0.0f, ceiling_db - input_level_db);
    } else {
        // Soft limiting with knee
        float margin = 3.0f; // 3dB margin before limiting starts
        if (input_level_db <= ceiling_db - margin) {
            return 0.0f;
        } else {
            return ceiling_db - input_level_db;
        }
    }
}

void Limiter::apply_brickwall_limiting(const float* input, float* output, uint32_t frame_count) {
    float ceiling_linear = dynamic_range_utils::dbfs_to_linear(ceiling_dbfs_);

    for (uint32_t i = 0; i < frame_count * channels_; ++i) {
        if (std::abs(input[i]) > ceiling_linear) {
            output[i] = (input[i] > 0.0f) ? ceiling_linear : -ceiling_linear;
        } else {
            output[i] = input[i];
        }
    }
}

void Limiter::update_attack_release_coefficients() {
    attack_coefficient_ = dynamic_range_utils::attack_time_to_coefficient(
        params_.attack_time_ms, sample_rate_);
    release_coefficient_ = dynamic_range_utils::release_time_to_coefficient(
        params_.release_time_ms, sample_rate_);
}

// MultiBandCompressor implementation (simplified)
MultiBandCompressor::MultiBandCompressor()
    : type_(DynamicRangeType::MULTIBAND_COMP)
    , model_(CompressionModel::DIGITAL)
    , bypassed_(false)
    , enabled_(true)
    , sample_rate_(44100)
    , channels_(2)
    , max_frame_size_(4096) {
}

bool MultiBandCompressor::initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) {
    sample_rate_ = sample_rate;
    channels_ = channels;
    max_frame_size_ = max_frame_size;

    params_ = DynamicRangeParameters();
    params_.type = DynamicRangeType::MULTIBAND_COMP;
    params_.num_bands = 3;

    initialize_bands();

    uint32_t total_samples = max_frame_size * channels_;
    input_buffer_.resize(total_samples, 0.0f);
    output_buffer_.resize(total_samples, 0.0f);
    band_output_buffer_.resize(total_samples, 0.0f);

    return true;
}

void MultiBandCompressor::shutdown() {
    bands_.clear();
    input_buffer_.clear();
    output_buffer_.clear();
    band_output_buffer_.clear();
}

bool MultiBandCompressor::reset() {
    for (auto& band : bands_) {
        if (band.compressor) {
            band.compressor->reset();
        }
    }
    return true;
}

bool MultiBandCompressor::process(const float* input, float* output, uint32_t frame_count) {
    if (bypassed_ || !input || !output || frame_count == 0) {
        if (input && output && frame_count > 0) {
            std::memcpy(output, input, frame_count * channels_ * sizeof(float));
        }
        return true;
    }

    std::memcpy(input_buffer_.data(), input, frame_count * channels_ * sizeof(float));

    // Process each band
    for (size_t band_idx = 0; band_idx < bands_.size(); ++band_idx) {
        process_band(input_buffer_.data(), band_output_buffer_.data(), frame_count, band_idx);
    }

    // Combine bands
    combine_bands(frame_count);

    // Copy to output
    std::memcpy(output, output_buffer_.data(), frame_count * channels_ * sizeof(float));

    return true;
}

bool MultiBandCompressor::process_interleaved(const float* input, float* output, uint32_t frame_count) {
    return process(input, output, frame_count);
}

bool MultiBandCompressor::process_with_sidechain(const float* input, const float* sidechain,
                                                 float* output, uint32_t frame_count) {
    return process(input, output, frame_count);
}

bool MultiBandCompressor::set_parameters(const DynamicRangeParameters& params) {
    params_ = params;
    return true;
}

DynamicRangeParameters MultiBandCompressor::get_parameters() const {
    return params_;
}

bool MultiBandCompressor::set_parameter(const std::string& name, float value) {
    return true; // Placeholder
}

float MultiBandCompressor::get_parameter(const std::string& name) const {
    return 0.0f; // Placeholder
}

bool MultiBandCompressor::set_threshold(float threshold_dbfs) {
    params_.threshold_dbfs = threshold_dbfs;
    return true;
}

bool MultiBandCompressor::set_ratio(float ratio) {
    params_.ratio = ratio;
    return true;
}

bool MultiBandCompressor::set_attack_time(float attack_ms) {
    params_.attack_time_ms = attack_ms;
    return true;
}

bool MultiBandCompressor::set_release_time(float release_ms) {
    params_.release_time_ms = release_ms;
    return true;
}

bool MultiBandCompressor::set_makeup_gain(float gain_db) {
    params_.makeup_gain_db = gain_db;
    return true;
}

bool MultiBandCompressor::set_knee_width(float width_db) {
    params_.knee_width_db = width_db;
    return true;
}

bool MultiBandCompressor::set_bypass(bool bypass) {
    bypassed_ = bypass;
    return true;
}

bool MultiBandCompressor::is_bypassed() const {
    return bypassed_;
}

bool MultiBandCompressor::set_enabled(bool enabled) {
    enabled_ = enabled;
    return true;
}

bool MultiBandCompressor::is_enabled() const {
    return enabled_;
}

bool MultiBandCompressor::save_preset(const std::string& name) {
    return true; // Placeholder
}

bool MultiBandCompressor::load_preset(const std::string& name) {
    return true; // Placeholder
}

std::vector<std::string> MultiBandCompressor::get_available_presets() const {
    return {"default", "vocal", "drums", "mastering"};
}

DynamicRangeType MultiBandCompressor::get_type() const {
    return type_;
}

CompressionModel MultiBandCompressor::get_model() const {
    return model_;
}

std::string MultiBandCompressor::get_name() const {
    return "Multi-band Compressor";
}

std::string MultiBandCompressor::get_version() const {
    return "1.0.0";
}

std::string MultiBandCompressor::get_description() const {
    return "Professional multi-band compressor with independent band control";
}

DynamicRangeStatistics MultiBandCompressor::get_statistics() const {
    return DynamicRangeStatistics(); // Placeholder
}

void MultiBandCompressor::reset_statistics() {
    // Statistics reset would go here
}

bool MultiBandCompressor::supports_real_time_parameter_changes() const {
    return true;
}

bool MultiBandCompressor::supports_gpu_acceleration() const {
    return false;
}

bool MultiBandCompressor::is_multiband() const {
    return true;
}

bool MultiBandCompressor::has_sidechain() const {
    return true;
}

double MultiBandCompressor::get_expected_latency_ms() const {
    return params_.attack_time_ms / 10.0;
}

float MultiBandCompressor::get_current_gain_reduction() const {
    return 0.0f; // Placeholder
}

float MultiBandCompressor::get_input_level() const {
    return 0.0f; // Placeholder
}

float MultiBandCompressor::get_output_level() const {
    return 0.0f; // Placeholder
}

float MultiBandCompressor::get_envelope_follower_value() const {
    return 0.0f; // Placeholder
}

bool MultiBandCompressor::is_clipping() const {
    return false;
}

void MultiBandCompressor::initialize_bands() {
    bands_.resize(params_.num_bands);

    for (size_t i = 0; i < bands_.size(); ++i) {
        bands_[i].compressor = std::make_unique<Compressor>();
        bands_[i].compressor->initialize(sample_rate_, channels_, max_frame_size_);
        bands_[i].frequency_hz = (i < params_.band_frequencies_hz.size()) ?
            params_.band_frequencies_hz[i] : 1000.0f;
        bands_[i].current_gain_reduction = 0.0f;
        bands_[i].enabled = true;
        bands_[i].band_buffer.resize(max_frame_size_ * channels_, 0.0f);
    }
}

void MultiBandCompressor::process_band(const float* input, float* output, uint32_t frame_count,
                                         uint32_t band_index) {
    if (band_index >= bands_.size()) {
        return;
    }

    auto& band = bands_[band_index];
    if (!band.enabled || !band.compressor) {
        std::memcpy(output, input, frame_count * channels_ * sizeof(float));
        return;
    }

    // Apply crossover filtering (simplified)
    for (uint32_t ch = 0; ch < channels_; ++ch) {
        for (uint32_t frame = 0; frame < frame_count; ++frame) {
            uint32_t sample_index = frame * channels_ + ch;
            float input_sample = input[sample_index];
            float filtered_sample = input_sample; // Placeholder - would apply actual crossover filter

            band.band_buffer[frame * channels_ + ch] = filtered_sample;
        }
    }

    // Process band with compressor
    band.compressor->process(band.band_buffer.data(), band.band_buffer.data(), frame_count);

    // Store result
    std::memcpy(output, band.band_buffer.data(), frame_count * channels_ * sizeof(float));
}

void MultiBandCompressor::combine_bands(uint32_t frame_count) {
    // Simple addition of bands (in real implementation would use proper crossover reconstruction)
    std::fill(output_buffer_.begin(), output_buffer_.begin() + frame_count * channels_, 0.0f);

    for (const auto& band : bands_) {
        if (band.enabled) {
            for (uint32_t i = 0; i < frame_count * channels_; ++i) {
                output_buffer_[i] += band.band_buffer[i];
            }
        }
    }
}

// Factory implementations
std::unique_ptr<DynamicRangeProcessor> DynamicRangeEffectsFactory::create_compressor() {
    return std::make_unique<Compressor>();
}

std::unique_ptr<DynamicRangeProcessor> DynamicRangeEffectsFactory::create_limiter() {
    return std::make_unique<Limiter>();
}

std::unique_ptr<DynamicRangeProcessor> DynamicRangeEffectsFactory::create_multiband_compressor(uint32_t bands) {
    auto mbc = std::make_unique<MultiBandCompressor>();
    DynamicRangeParameters params;
    params.num_bands = bands;
    mbc->set_parameters(params);
    return std::move(mbc);
}

std::unique_ptr<DynamicRangeProcessor> DynamicRangeEffectsFactory::create_vocal_compressor() {
    auto comp = std::make_unique<Compressor>();
    DynamicRangeParameters params;
    params.threshold_dbfs = -18.0f;
    params.ratio = 3.0f;
    params.attack_time_ms = 3.0f;
    params.release_time_ms = 150.0f;
    params.knee_width_db = 4.0f;
    comp->set_parameters(params);
    return std::move(comp);
}

std::unique_ptr<DynamicRangeProcessor> DynamicRangeEffectsFactory::create_mastering_compressor() {
    auto comp = std::make_unique<Compressor>();
    DynamicRangeParameters params;
    params.threshold_dbfs = -8.0f;
    params.ratio = 2.0f;
    params.attack_time_ms = 10.0f;
    params.release_time_ms = 500.0f;
    params.knee_width_db = 6.0f;
    comp->set_parameters(params);
    return std::move(comp);
}

std::unique_ptr<DynamicRangeProcessor> DynamicRangeEffectsFactory::create_noise_gate() {
    auto comp = std::make_unique<Compressor>();
    DynamicRangeParameters params;
    params.type = DynamicRangeType::GATE;
    params.threshold_dbfs = -40.0f;
    params.ratio = 10.0f;
    params.attack_time_ms = 1.0f;
    params.release_time_ms = 200.0f;
    params.knee_width_db = 2.0f;
    comp->set_parameters(params);
    return std::move(comp);
}

std::unique_ptr<DynamicRangeProcessor> DynamicRangeEffectsFactory::create_deesser() {
    auto comp = std::make_unique<Compressor>();
    DynamicRangeParameters params;
    params.type = DynamicRangeType::DEESSER;
    params.threshold_dbfs = -6.0f;
    params.ratio = 4.0f;
    params.attack_time_ms = 1.0f;
    params.release_time_ms = 100.0f;
    params.knee_width_db = 1.0f;
    params.enable_sidechain = true;
    params.sidechain_freq_hz = 4000.0f;
    comp->set_parameters(params);
    return std::move(comp);
}

std::vector<std::string> DynamicRangeEffectsFactory::get_available_effect_types() {
    return {
        "compressor", "limiter", "expander", "gate", "deesser",
        "multiband_comp", "upward_comp", "parallel_comp", "transient_shaper",
        "auto_gain", "ducker", "envelope_follower", "rms_compressor",
        "peak_compressor", "smooth_limiter", "brickwall_limiter"
    };
}

// Utility namespace implementations
namespace dynamic_range_utils {

float linear_to_db(float linear) {
    return linear > 0.0f ? 20.0f * std::log10(linear) : -INFINITY;
}

float db_to_linear(float db) {
    return std::pow(10.0f, db / 20.0f);
}

float dbfs_to_linear(float dbfs) {
    return db_to_linear(dbfs);
}

float linear_to_dbfs(float linear) {
    return linear_to_db(linear);
}

float rms_to_db(float rms, uint32_t num_samples) {
    return rms > 0.0f ? linear_to_db(rms) : -INFINITY;
}

float peak_to_db(float peak) {
    return peak > 0.0f ? linear_to_db(peak) : -INFINITY;
}

float time_constant_to_coefficient(float time_ms, uint32_t sample_rate) {
    return std::exp(-1000.0f / (time_ms * sample_rate));
}

float coefficient_to_time_constant(float coefficient, uint32_t sample_rate) {
    return -1000.0f / (sample_rate * std::log(coefficient));
}

float attack_time_to_coefficient(float attack_ms, uint32_t sample_rate) {
    return time_constant_to_coefficient(attack_ms, sample_rate);
}

float release_time_to_coefficient(float release_ms, uint32_t sample_rate) {
    return time_constant_to_coefficient(release_ms, sample_rate);
}

float calculate_compression_gain(float input_db, float threshold_db, float ratio, float knee_width) {
    if (input_db <= threshold_db - knee_width / 2.0f) {
        return 0.0f;
    }

    float over_threshold = input_db - threshold_db;
    if (ratio > 1.0f) {
        return over_threshold * (1.0f - 1.0f / ratio);
    }

    return 0.0f;
}

float calculate_expansion_gain(float input_db, float threshold_db, float ratio, float range_db) {
    if (input_db < threshold_db) {
        float under_threshold = threshold_db - input_db;
        return -under_threshold * (ratio - 1.0f);
    }
    return 0.0f;
}

float calculate_limiter_gain(float input_db, float ceiling_db) {
    return std::min(0.0f, ceiling_db - input_db);
}

float calculate_gate_gain(float input_db, float threshold_db, float range_db, float hysteresis) {
    if (input_db < threshold_db - hysteresis) {
        float under_threshold = threshold_db - input_db;
        return std::min(range_db, under_threshold);
    } else if (input_db > threshold_db + hysteresis) {
        return 0.0f;
    }
    return 0.0f;
}

float soft_knee(float input_db, float threshold_db, float knee_width) {
    float knee_start = threshold_db - knee_width / 2.0f;
    float knee_end = threshold_db + knee_width / 2.0f;

    if (input_db < knee_start) {
        return 0.0f;
    } else if (input_db > knee_end) {
        return input_db - threshold_db;
    } else {
        // Soft knee region - quadratic interpolation
        float knee_ratio = 1.0f + 3.0f * std::pow((input_db - knee_start) / knee_width - 0.5, 2);
        return (input_db - threshold_db) * (1.0f - 1.0f / knee_ratio);
    }
}

float calculate_rms(const float* buffer, uint32_t frame_count) {
    float sum_squares = 0.0f;
    for (uint32_t i = 0; i < frame_count; ++i) {
        sum_squares += buffer[i] * buffer[i];
    }
    return std::sqrt(sum_squares / frame_count);
}

float calculate_peak(const float* buffer, uint32_t frame_count) {
    float peak = 0.0f;
    for (uint32_t i = 0; i < frame_count; ++i) {
        peak = std::max(peak, std::abs(buffer[i]));
    }
    return peak;
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

} // namespace dynamic_range_utils

} // namespace vortex::core::dsp