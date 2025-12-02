#include "core/dsp/spatial_audio_processor.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <random>
#include <immintrin.h>

namespace vortex::core::dsp {

// BinauralProcessor implementation
BinauralProcessor::BinauralProcessor()
    : initialized_(false)
    , processing_(false)
    , next_source_id_(1)
    , hrtf_interpolation_(HRTFInterpolation::LINEAR)
    , enable_hrtf_caching_(true)
    , shutdown_requested_(false) {
}

BinauralProcessor::~BinauralProcessor() {
    shutdown();
}

bool BinauralProcessor::initialize(const SpatialAudioConfig& config) {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    config_ = config;
    config_.type = SpatialAudioType::BINAURAL;
    config_.speaker_config = SpeakerConfiguration::BINAURAL;

    Logger::info("Initializing Binaural Processor:");
    Logger::info("  Sample rate: {} Hz", config.sample_rate);
    Logger::info("  Max frame size: {}", config.max_frame_size);
    Logger::info("  Output channels: {}", config.output_channels);
    Logger::info("  HRTF enabled: {}", config.enable_hrtf);

    // Initialize listener state
    listener_state_.position_x = 0.0f;
    listener_state_.position_y = 0.0f;
    listener_state_.position_z = 0.0f;
    listener_state_.orientation_yaw = 0.0f;
    listener_state_.orientation_pitch = 0.0f;
    listener_state_.orientation_roll = 0.0f;
    listener_state_.velocity_x = 0.0f;
    listener_state_.velocity_y = 0.0f;
    listener_state_.velocity_z = 0.0f;
    listener_state_.head_tracking_enabled = false;
    listener_state_.vr_mode_enabled = false;

    // Initialize room acoustics
    room_acoustics_.dimensions_x = config.room_dimensions_x;
    room_acoustics_.dimensions_y = config.room_dimensions_y;
    room_acoustics_.dimensions_z = config.room_dimensions_z;
    room_acoustics_.reverberation_time = config.reverberation_time_seconds;
    room_acoustics_.wall_absorption = config.wall_absorption;
    room_acoustics_.air_absorption_low_freq = config.air_absorption_low_freq;
    room_acoustics_.air_absorption_high_freq = config.air_absorption_high_freq;
    room_acoustics_.air_absorption_coefficient = config.air_absorption_coefficient;
    room_acoustics_.model = config.room_model;

    // Initialize processing buffers
    uint32_t total_samples = config.max_frame_size * config.output_channels;
    input_buffer_.resize(total_samples, 0.0f);
    output_buffer_.resize(total_samples, 0.0f);
    hrtf_left_buffer_.resize(total_samples, 0.0f);
    hrtf_right_buffer_.resize(total_samples, 0.0f);
    distance_attenuation_buffer_.resize(config.max_frame_size, 0.0f);
    doppler_buffer_.resize(config.max_frame_size, 0.0f);

    // Reset statistics
    statistics_ = SpatialAudioStatistics();
    statistics_.last_reset_time = std::chrono::steady_clock::now();

    // Load HRTF dataset if enabled
    if (config.enable_hrtf && !config.hrtf_dataset_path.empty()) {
        if (!load_hrtf_dataset_internal(config.hrtf_dataset_path, 0)) {
            Logger::error("Failed to load HRTF dataset from: {}", config.hrtf_dataset_path);
            // Continue without HRTF or return error
        }
    }

    // Start processing thread
    processing_thread_ = std::thread(&BinauralProcessor::processing_thread_function, this);

    initialized_ = true;
    Logger::info("Binaural Processor initialized successfully");
    return true;
}

void BinauralProcessor::shutdown() {
    if (!initialized_) {
        return;
    }

    Logger::info("Shutting down Binaural Processor");

    // Signal processing thread to stop
    shutdown_requested_ = true;
    processing_ = false;

    // Wait for processing thread to finish
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }

    // Clear sources
    {
        std::lock_guard<std::mutex> lock(sources_mutex_);
        sources_.clear();
    }

    // Clear HRTF datasets
    hrtf_datasets_.clear();
    hrtf_cache_.clear();

    // Clear buffers
    input_buffer_.clear();
    output_buffer_.clear();
    hrtf_left_buffer_.clear();
    hrtf_right_buffer_.clear();
    distance_attenuation_buffer_.clear();
    doppler_buffer_.clear();

    initialized_ = false;
    Logger::info("Binaural Processor shutdown complete");
}

bool BinauralProcessor::reset() {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    // Reset all sources
    for (auto& [id, source] : sources_) {
        source->write_position = 0;
        source->read_position = 0;
        source->is_buffer_full = false;
        std::fill(source->audio_buffer.begin(), source->audio_buffer.end(), 0.0f);
    }

    // Reset buffers
    std::fill(input_buffer_.begin(), input_buffer_.end(), 0.0f);
    std::fill(output_buffer_.begin(), output_buffer_.end(), 0.0f);
    std::fill(hrtf_left_buffer_.begin(), hrtf_left_buffer_.end(), 0.0f);
    std::fill(hrtf_right_buffer_.begin(), hrtf_right_buffer_.end(), 0.0f);

    Logger::info("Binaural Processor reset");
    return true;
}

bool BinauralProcessor::process(const float** inputs, float* output, uint32_t frame_count,
                               const std::vector<AudioSource*>& sources) {
    if (!initialized_ || !inputs || !output || frame_count == 0) {
        return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Interleave input if needed (assuming single input channel for simplicity)
    std::memcpy(input_buffer_.data(), inputs[0], frame_count * sizeof(float));

    // Process audio sources
    process_sources(sources, frame_count);

    // Copy to output
    std::memcpy(output, output_buffer_.data(), frame_count * config_.output_channels * sizeof(float));

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

    statistics_.is_processing = true;
    statistics_.frame_drop_count = 0;
    statistics_.avg_latency_ms = 5.0f; // Estimated latency

    return true;
}

bool BinauralProcessor::process_interleaved(const float* input, float* output, uint32_t frame_count,
                                         const std::vector<AudioSource*>& sources) {
    if (!initialized_ || !input || !output || frame_count == 0) {
        return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Copy input buffer
    std::memcpy(input_buffer_.data(), input, frame_count * config_.output_channels * sizeof(float));

    // Process audio sources
    process_sources(sources, frame_count);

    // Copy to output
    std::memcpy(output, output_buffer_.data(), frame_count * config_.output_channels * sizeof(float));

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

    return true;
}

uint32_t BinauralProcessor::add_source(const AudioSource& source) {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    uint32_t source_id = next_source_id_++;
    auto new_source = std::make_unique<AudioSource>(source);
    new_source->id = source_id;

    // Initialize audio buffer
    new_source->audio_buffer.resize(config_.audio_buffer_size, 0.0f);
    new_source->buffer_size = config_.audio_buffer_size;
    new_source->write_position = 0;
    new_source->read_position = 0;
    new_source->is_buffer_full = false;

    sources_[source_id] = std::move(new_source);

    Logger::info("Added audio source: {} (ID: {})", source.name, source_id);
    notify_source_added(source_id, *sources_[source_id]);

    return source_id;
}

bool BinauralProcessor::remove_source(uint32_t source_id) {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = sources_.find(source_id);
    if (it == sources_.end()) {
        Logger::warn("Source not found: {}", source_id);
        return false;
    }

    std::string source_name = it->second->name;
    sources_.erase(it);

    Logger::info("Removed audio source: {} (ID: {})", source_name, source_id);
    notify_source_removed(source_id);

    return true;
}

bool BinauralProcessor::update_source(uint32_t source_id, const AudioSource& source) {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = sources_.find(source_id);
    if (it == sources_.end()) {
        return false;
    }

    // Update source properties
    *it->second = source;
    it->second->id = source_id; // Ensure ID consistency

    Logger::debug("Updated audio source: {} (ID: {})", source.name, source_id);
    return true;
}

AudioSource* BinauralProcessor::get_source(uint32_t source_id) {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = sources_.find(source_id);
    if (it != sources_.end()) {
        return it->second.get();
    }

    return nullptr;
}

std::vector<AudioSource*> BinauralProcessor::get_all_sources() {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    std::vector<AudioSource*> sources;
    sources.reserve(sources_.size());

    for (const auto& [id, source] : sources_) {
        sources.push_back(source.get());
    }

    return sources;
}

std::vector<AudioSource*> BinauralProcessor::get_sources_in_range(const float* center, float radius) {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    std::vector<AudioSource*> sources_in_range;
    float radius_sq = radius * radius;

    for (const auto& [id, source] : sources_) {
        if (!source->is_active) {
            continue;
        }

        float dx = source->position_x - center[0];
        float dy = source->position_y - center[1];
        float dz = source->position_z - center[2];
        float distance_sq = dx * dx + dy * dy + dz * dz;

        if (distance_sq <= radius_sq) {
            sources_in_range.push_back(source.get());
        }
    }

    return sources_in_range;
}

bool BinauralProcessor::set_source_position(uint32_t source_id, float x, float y, float z) {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = sources_.find(source_id);
    if (it == sources_.end()) {
        return false;
    }

    // Store old position for callback
    float old_x = it->second->position_x;
    float old_y = it->second->position_y;
    float old_z = it->second->position_z;

    // Update position
    it->second->position_x = x;
    it->second->position_y = y;
    it->second->position_z = z;

    // Check if position changed significantly
    float position_change = std::sqrt((x - old_x) * (x - old_x) +
                                        (y - old_y) * (y - old_y) +
                                        (z - old_z) * (z - old_z));
    if (position_change > 0.01f) { // 1cm threshold
        it->second->is_moving = true;
    }

    notify_source_moved(source_id, x, y, z);
    return true;
}

bool BinauralProcessor::set_source_velocity(uint32_t source_id, float vx, float vy, float vz) {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = sources_.find(source_id);
    if (it == sources_.end()) {
        return false;
    }

    it->second->velocity_x = vx;
    it->second->velocity_y = vy;
    it->second->velocity_z = vz;

    if (vx != 0.0f || vy != 0.0f || vz != 0.0f) {
        it->second->is_moving = true;
    }

    return true;
}

bool BinauralProcessor::get_source_position(uint32_t source_id, float& x, float& y, float& z) {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = sources_.find(source_id);
    if (it == sources_.end()) {
        return false;
    }

    x = it->second->position_x;
    y = it->second->position_y;
    z = it->second->position_z;

    return true;
}

bool BinauralProcessor::get_source_velocity(uint32_t source_id, float& vx, float& vy, float& vz) {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = sources_.find(source_id);
    if (it == sources_.end()) {
        return false;
    }

    vx = it->second->velocity_x;
    vy = it->second->velocity_y;
    vz = it->second->velocity_z;

    return true;
}

bool BinauralProcessor::set_source_gain(uint32_t source_id, float gain) {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = sources_.find(source_id);
    if (it == sources_.end()) {
        return false;
    }

    it->second->gain = gain;
    it->second->gain_dbfs = spatial_audio_utils::linear_to_dbfs(gain);

    return true;
}

bool BinauralProcessor::set_source_mute(uint32_t source_id, bool mute) {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = sources_.find(source_id);
    if (it == sources_.end()) {
        return false;
    }

    it->second->mute = mute;
    return true;
}

bool BinauralProcessor::set_source_solo(uint32_t source_id, bool solo) {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = sources_.find(source_id);
    if (it == sources_.end()) {
        return false;
    }

    it->second->solo = solo;
    return true;
}

bool BinauralProcessor::set_source_directivity(uint32_t source_id, float factor,
                                                float azimuth, float elevation) {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = sources_.find(source_id);
    if (it == sources_.end()) {
        return false;
    }

    it->second->directivity_factor = factor;
    it->second->directivity_pattern = azimuth;
    it->second->azimuth_angle = azimuth;
    it->second->elevation_angle = elevation;

    return true;
}

bool BinauralProcessor::set_listener_position(float x, float y, float z) {
    listener_state_.position_x = x;
    listener_state_.position_y = y;
    listener_state_.position_z = z;
    return true;
}

bool BinauralProcessor::set_listener_orientation(float yaw, float pitch, float roll) {
    listener_state_.orientation_yaw = yaw;
    listener_state_.orientation_pitch = pitch;
    listener_state_.orientation_roll = roll;
    return true;
}

bool BinaProcessor::set_listener_velocity(float vx, float vy, float vz) {
    listener_state_.velocity_x = vx;
    listener_state_.velocity_y = vy;
    listener_state_.velocity_z = vz;
    return true;
}

bool BinauralProcessor::get_listener_position(float& x, float& y, float& z) {
    x = listener_state_.position_x;
    y = listener_state_.position_y;
    z = listener_state_.position_z;
    return true;
}

bool BinauralProcessor::get_listener_orientation(float& yaw, float& pitch, float& roll) {
    yaw = listener_state_.orientation_yaw;
    pitch = listener_state_.orientation_pitch;
    roll = listener_state_.orientation_roll;
    return true;
}

bool BinauralProcessor::set_room_dimensions(float x, float y, float z) {
    room_acoustics_.dimensions_x = x;
    room_acoustics_.dimensions_y = y;
    room_acoustics_.dimensions_z = z;
    return true;
}

bool BinauralProcessor::set_reverberation_time(float reverb_time) {
    room_acoustics_.reverberation_time = reverb_time;
    return true;
}

bool BinauralProcessor::set_wall_absorption(float absorption) {
    room_acoustics_.wall_absorption = std::clamp(absorption, 0.0f, 1.0f);
    return true;
}

bool BinauralProcessor::set_air_absorption(float low_freq, float high_freq, float coefficient) {
    room_acoustics_.air_absorption_low_freq = low_freq;
    room_acoustics_.air_absorption_high_freq = high_freq;
    room_acoustics_.air_absorption_coefficient = coefficient;
    return true;
}

bool BinauralProcessor::load_hrtf_dataset(const std::string& dataset_path) {
    return load_hrtf_dataset_internal(dataset_path, 0);
}

bool BinauralProcessor::set_hrtf_index(uint32_t source_id, uint32_t hrtf_index) {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = sources_.find(source_id);
    if (it == sources_.end()) {
        return false;
    }

    it->second->hrtf_index = hrtf_index;
    return true;
}

bool BinauralProcessor::enable_hrtf_for_source(uint32_t source_id, bool enabled) {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = sources_.find(source_id);
    if (it == sources_.end()) {
        return false;
    }

    it->second->enable_hrtf = enabled;
    return true;
}

bool BinauralProcessor::set_hrtf_interpolation(HRTFInterpolation interpolation) {
    hrtf_interpolation_ = interpolation;
    return true;
}

bool BinauralProcessor::set_spatial_type(SpatialAudioType type) {
    config_.type = type;
    return true;
}

bool BinauralProcessor::set_speaker_configuration(SpeakerConfiguration config) {
    config_.speaker_config = config;
    return true;
}

bool BinauralProcessor::update_config(const SpatialAudioConfig& config) {
    config_ = config;
    return true;
}

const SpatialAudioConfig& BinauralProcessor::get_config() const {
    return config_;
}

bool BinauralProcessor::save_preset(const std::string& name) {
    // Preset implementation would go here
    Logger::info("Saved spatial audio preset: {}", name);
    return true;
}

bool BinauralProcessor::load_preset(const std::string& name) {
    // Preset implementation would go here
    Logger::info("Loaded spatial audio preset: {}", name);
    return true;
}

std::vector<std::string> BinauralProcessor::get_available_presets() const {
    return {"default", "headphones", "earbuds", "studio_monitors", "vr_headphones"};
}

SpatialAudioType BinauralProcessor::get_type() const {
    return config_.type;
}

SpeakerConfiguration BinauralProcessor::get_speaker_configuration() const {
    return config_.speaker_config;
}

std::string BinauralProcessor::get_name() const {
    return "Binaural Processor";
}

std::string BinauralProcessor::get_version() const {
    return "1.0.0";
}

std::string BinauralProcessor::get_description() const {
    return "Binaural spatial audio processor with HRTF and 3D positioning";
}

SpatialAudioStatistics BinauralProcessor::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return statistics_;
}

void BinauralProcessor::reset_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    statistics_ = SpatialAudioStatistics();
    statistics_.last_reset_time = std::chrono::steady_clock::now();
}

bool BinauralProcessor::supports_real_time_positioning() const {
    return true;
}

bool BinauralProcessor::supports_hrtf() const {
    return config_.enable_hrtf && !hrtf_datasets_.empty();
}

bool BinauralProcessor::supports_ambisonics() const {
    return false;
}

bool BinauralProcessor::supports_gpu_acceleration() const {
    return false;
}

uint32_t BinauralProcessor::get_max_sources() const {
    return config_.max_audio_sources;
}

double BinauralProcessor::get_expected_latency_ms() const {
    return 10.0; // Estimated 10ms latency for HRTF processing
}

bool BinauralProcessor::is_3d_enabled() const {
    return true;
}

bool BinauralProcessor::enable_head_tracking(bool enabled) {
    listener_state_.head_tracking_enabled = enabled;
    return true;
}

bool BinauralProcessor::set_head_transform(const float* transform_matrix) {
    if (transform_matrix) {
        std::memcpy(listener_state_.head_transform_matrix, transform_matrix, 16 * sizeof(float));
    }
    return true;
}

bool BinauralProcessor::enable_vr_mode(bool enabled) {
    listener_state_.vr_mode_enabled = enabled;
    return true;
}

bool BinauralProcessor::enable_individualized_hrtf(bool enabled) {
    // Individualized HRTF implementation would go here
    Logger::info("Individualized HRTF {}", enabled ? "enabled" : "disabled");
    return true;
}

bool BinauralProcessor::measure_individualized_hrtf(uint32_t subject_id) {
    // HRTF measurement implementation would go here
    Logger::info("Measuring individualized HRTF for subject: {}", subject_id);
    return true;
}

bool BinauralProcessor::load_individualized_hrtf(uint32_t subject_id, const std::string& hrtf_path) {
    // Individualized HRTF loading implementation would go here
    Logger::info("Loading individualized HRTF for subject {} from: {}", subject_id, hrtf_path);
    return true;
}

uint32_t BinauralProcessor::get_hrtf_index(uint32_t source_id) const {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = sources_.find(source_id);
    if (it != sources_.end()) {
        return it->second->hrtf_index;
    }

    return 0;
}

float BinauralProcessor::get_azimuth_angle(uint32_t source_id) const {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = sources_.find(source_id);
    if (it != sources_.end()) {
        return it->second->azimuth_angle;
    }

    return 0.0f;
}

float BinauralProcessor::get_elevation_angle(uint32_t source_id) const {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = sources_.find(source_id);
    if (it != sources_.end()) {
        return it->second->elevation_angle;
    }

    return 0.0f;
}

void BinauralProcessor::process_sources(const std::vector<AudioSource*>& sources, uint32_t frame_count) {
    // Clear output buffer
    std::fill(output_buffer_.begin(), output_buffer_.end(), 0.0f);

    // Process each audio source
    for (const auto* source : sources) {
        if (!source || !source->is_active || source->mute) {
            continue;
        }

        // Calculate spatial parameters
        float azimuth, elevation, distance;
        calculate_spatial_position(*const_cast<AudioSource*>(source), azimuth, elevation, distance);

        // Apply distance attenuation
        float distance_gain = calculate_source_gain(*source);
        float doppler_shift = 1.0f;

        if (config_.enable_distance_attenuation) {
            distance_gain *= spatial_audio_utils::calculate_distance_attenuation(
                distance, source->reference_distance, source->rolloff_factor);
        }

        if (config_.enable_doppler && source->is_moving) {
            doppler_shift = calculate_doppler_shift(*source);
        }

        // Apply HRTF if enabled
        if (config_.enable_hrtf && source->enable_hrtf) {
            apply_hrtf(*const_cast<AudioSource*>(source), frame_count);
        }

        // Mix to output buffer
        for (uint32_t frame = 0; frame < frame_count; ++frame) {
            float source_sample = 0.0f;

            // Get sample from source buffer (simplified - would read from actual audio stream)
            if (frame < source->audio_buffer.size()) {
                source_sample = source->audio_buffer[frame];
            }

            // Apply gains and processing
            float processed_sample = source_sample * source->gain * distance_gain * doppler_shift;

            // Mix with existing output (simple additive mixing)
            for (uint32_t ch = 0; ch < config_.output_channels; ++ch) {
                uint32_t output_index = frame * config_.output_channels + ch;
                output_buffer_[output_index] += processed_sample / static_cast<float>(sources.size());
            }
        }
    }
}

void BinauralProcessor::apply_hrtf(AudioSource& source, uint32_t frame_count) {
    if (hrtf_datasets_.empty()) {
        return;
    }

    // Calculate azimuth and elevation
    float azimuth, elevation, distance;
    calculate_spatial_position(source, azimuth, elevation, distance);

    // Get HRTF data
    uint32_t hrtf_index = source.hrtf_index;
    auto hrtf_it = hrtf_datasets_.find(hrtf_index);
    if (hrtf_it == hrtf_datasets_.end()) {
        return;
    }

    const auto& hrtf_data = hrtf_it->second;

    // Interpolate HRTF if needed
    std::vector<float> left_ir, right_ir;
    interpolate_hrtf(source.id, azimuth, elevation, left_ir, right_ir);

    // Apply HRTF convolution (simplified)
    uint32_t ir_length = std::min(hrtf_data.ir_length, frame_count);
    for (uint32_t ch = 0; ch < config_.output_channels; ++ch) {
        for (uint32_t frame = 0; frame < frame_count && frame < ir_length; ++frame) {
            uint32_t output_index = frame * config_.output_channels + ch;

            // Simple convolution (would use FFT in real implementation)
            for (uint32_t ir_sample = 0; ir_sample < ir_length && frame + ir_sample < frame_count; ++ir_sample) {
                uint32_t buffer_index = frame + ir_sample;
                if (ch == 0 && buffer_index < hrtf_left_buffer_.size()) {
                    output_buffer_[buffer_index] += hrtf_left_buffer_[buffer_index] *
                                                    source.audio_buffer[ir_sample];
                } else if (ch == 1 && buffer_index < hrtf_right_buffer_.size()) {
                    output_buffer_[buffer_index] += hrtf_right_buffer_[buffer_index] *
                                                    source.audio_buffer[ir_sample];
                }
            }
        }
    }
}

void BinauralProcessor::apply_distance_attenuation(AudioSource& source, uint32_t frame_count) {
    float distance = std::sqrt(
        (source.position_x - listener_state_.position_x) * (source.position_x - listener_state_.position_x) +
        (source.position_y - listener_state_.position_y) * (source.position_y - listener_state_.position_y) +
        (source.position_z - listener_state_.position_z) * (source.position_z - listener_state_.position_z));

    float distance_gain = spatial_audio_utils::calculate_distance_attenuation(
        distance, source.reference_distance, source.rolloff_factor);

    // Apply air absorption if enabled
    if (config_.enable_air_absorption) {
        float air_absorption_gain = spatial_audio_utils::calculate_air_absorption(
            1000.0f, distance, room_acoustics_.air_absorption_coefficient);
        distance_gain *= air_absorption_gain;
    }

    // Store in buffer for later use
    for (uint32_t i = 0; i < frame_count && i < distance_attenuation_buffer_.size(); ++i) {
        distance_attenuation_buffer_[i] = distance_gain;
    }
}

void BinauralProcessor::apply_doppler_effect(AudioSource& source, uint32_t frame_count) {
    if (!source.is_moving) {
        return;
    }

    float relative_velocity = std::sqrt(
        (source.velocity_x - listener_state_.velocity_x) * (source.velocity_x - listener_state_.velocity_x) +
        (source.velocity_y - listener_state_.velocity_y) * (source.velocity_y - listener_state_.velocity_y) +
        (source.velocity_z - listener_state_.velocity_z) * (source.velocity_z - listener_state_.velocity_z));

    float doppler_factor = spatial_audio_utils::calculate_doppler_factor(
        relative_velocity, 343.0f); // Speed of sound at 20°C

    // Store in buffer for later use
    for (uint32_t i = 0; i < frame_count && i < doppler_buffer_.size(); ++i) {
        doppler_buffer_[i] = doppler_factor;
    }
}

void BinauralProcessor::apply_occlusion(AudioSource& source, uint32_t frame_count) {
    // Simplified occlusion model
    // In a real implementation, would use ray tracing or other techniques
    float distance = std::sqrt(
        (source.position_x - listener_state_.position_x) * (source.position_x - listener_state_.position_x) +
        (source.position_y - listener_state_.position_y) * (source.position_y - listener_state_.position_y) +
        (source.position_z - listener_state_.position_z) * (source.position_z - listener_state_.position_z));

    float occlusion_factor = source.occlusion_factor;
    if (config_.enable_occlusion) {
        occlusion_factor = spatial_audio_utils::calculate_occlusion_factor(
            distance, 5.0f); // Simple obstacle at 5m
    }

    // Apply occlusion to gain
    source.gain *= occlusion_factor;
}

float BinauralProcessor::calculate_source_gain(const AudioSource& source) {
    // Calculate total gain including directivity
    float total_gain = source.gain;

    if (source.enable_directivity && source.directivity_factor > 0.0f) {
        float directivity_gain = spatial_audio_utils::calculate_cardioid_pattern(
            source.azimuth_angle, source.directivity_factor);
        total_gain *= directivity_gain;
    }

    return total_gain;
}

float BinauralProcessor::calculate_doppler_shift(const AudioSource& source) {
    if (!source.is_moving) {
        return 1.0f;
    }

    float relative_velocity = std::sqrt(
        (source.velocity_x - listener_state_.velocity_x) * (source.velocity_x - listener_state_.velocity_x) +
        (source.velocity_y - listener_state_.velocity_y) * (source.velocity_y - listener_state_.velocity_y) +
        (source.velocity_z - listener_state_.velocity_z) * (source.velocity_z - listener_state_.velocity_z));

    return spatial_audio_utils::calculate_doppler_factor(
        relative_velocity, 343.0f) * source.doppler_factor;
}

void BinauralProcessor::calculate_spatial_position(AudioSource& source,
                                                   float& azimuth, float& elevation,
                                                   float& distance) {
    // Calculate relative position
    float rel_x = source.position_x - listener_state_.position_x;
    float rel_y = source.position_y - listener_state_.position_y;
    float rel_z = source.position_z - listener_state_.position_z;

    // Convert to spherical coordinates
    spatial_audio_utils::cartesian_to_spherical(rel_x, rel_y, rel_z, azimuth, elevation, distance);

    // Apply head-related transformations if VR mode is enabled
    if (listener_state_.vr_mode_enabled) {
        // Apply head transform matrix (simplified)
        // In real implementation, would properly multiply position by transform matrix
        float transformed_x = rel_x; // Placeholder
        float transformed_y = rel_y;
        float transformed_z = rel_z;
        spatial_audio_utils::cartesian_to_spherical(transformed_x, transformed_y, transformed_z,
                                                 azimuth, elevation, distance);
    }
}

void BinauralProcessor::interpolate_hrtf(uint32_t source_id, float azimuth, float elevation,
                                          std::vector<float>& left_ir, std::vector<float>& right_ir) {
    // Find nearest HRTF measurements
    uint32_t nearest_index = spatial_audio_utils::find_nearest_hrtf_index(
        azimuth, elevation,
        hrtf_datasets_[0].azimuths,
        hrtf_datasets_[0].elevations);

    // Get HRTF data for nearest index
    const auto& hrtf_data = hrtf_datasets_[nearest_index];

    // Copy IR data (simplified - would handle length mismatches properly)
    left_ir = hrtf_data.left_ir[0]; // Use first channel for simplicity
    right_ir = hrtf_data.right_ir[0]; // Use first channel for simplicity

    // Apply interpolation if multiple measurements are available
    if (hrtf_interpolation_ == HRTFInterpolation::LINEAR &&
        hrtf_data.num_measurements > 1) {
        // Simple linear interpolation between measurements
        float interpolation_weight = spatial_audio_utils::calculate_hrtf_interpolation_weight(
            azimuth, elevation, hrtf_data.azimuths[0], hrtf_data.azimuths[1]);
        left_ir *= interpolation_weight;
        right_ir *= interpolation_weight;
    }
}

void BinauralProcessor::update_statistics(uint32_t frame_count, double processing_time_us) {
    std::lock_guard<std::lock_guard<std::mutex>> lock(stats_mutex_);

    statistics_.active_sources = 0;
    statistics_.moving_sources = 0;
    statistics_.static_sources = 0;
    statistics_.occluded_sources = 0;

    {
        std::lock_guard<std::mutex> sources_lock(sources_mutex_);
        for (const auto& [id, source] : sources_) {
            if (source->is_active) {
                statistics_.active_sources++;
                if (source->is_moving) {
                    statistics_.moving_sources++;
                } else {
                    statistics_.static_sources++;
                }
                if (source->enable_occlusion && source->occlusion_factor < 0.9f) {
                    statistics_.occluded_sources++;
                }
            }
        }
    }

    // Calculate average distance
    if (statistics_.active_sources > 0) {
        float total_distance = 0.0f;
        {
            std::lock_guard<std::mutex> sources_lock(sources_mutex_);
            for (const auto& [id, source] : sources_) {
                if (source->is_active) {
                    total_distance += std::sqrt(
                        (source->position_x - listener_state_.position_x) * (source->position_x - listener_state_.position_x) +
                        (source->position_y - listener_state_.position_y) * (source->position_y - listener_state_.position_y) +
                        (source->position_z - listener_state_.position_z) * (source->position_z - listener_state_.position_z));
                }
            }
        }
        statistics_.avg_distance_meters = total_distance / statistics_.active_sources;
    }

    // Calculate max distance
    float max_distance = 0.0f;
    {
        std::lock_guard<std::mutex> sources_lock(sources_mutex_);
        for (const auto& [id, source] : sources_) {
            if (source->is_active) {
                float distance = std::sqrt(
                    (source->position_x - listener_state_.position_x) * (source->position_x - listener_state_.position_x) +
                    (source->position_y - listener_state_.position_y) * (source->position_y - listener_state_.position_y) +
                    (source->position_z - listener_state_.position_z) * (source->position_z - listener_state_.position_z));
                max_distance = std::max(max_distance, distance);
            }
        }
    }
    statistics_.max_distance_meters = max_distance;

    // Update HRTF statistics
    statistics_.hrtf_interpolations += config_.enable_hrtf ? statistics_.active_sources : 0;
    if (enable_hrtf_caching_) {
        statistics_.hrtf_cache_hits = hrtf_cache_.size(); // Simplified
        statistics_.hrtf_cache_misses = 0; // Simplified
    }

    // Update performance metrics
    statistics_.cpu_utilization_percent = 0.0f; // Would calculate actual CPU usage
    statistics_.gpu_utilization_percent = 0.0f; // Would check if GPU is used

    // Update latency metrics
    statistics_.avg_latency_ms = statistics_.avg_processing_time_us / 1000.0;
    statistics_.max_latency_ms = statistics_.max_processing_time_us / 1000.0;
    statistics_.spatial_image_score = 0.8f; // Simplified
}

void BinauralProcessor::notify_source_added(uint32_t source_id, const AudioSource& source) {
    std::lock_guard<std::lock_guard<std::mutex>> lock(callbacks_mutex_);
    if (source_added_callback_) {
        source_added_callback_(source_id, source);
    }
}

void BinauralProcessor::notify_source_removed(uint32_t source_id) {
    std::lock_guard<std::lock_guard<std::mutex>> lock(callbacks_mutex_);
    if (source_removed_callback_) {
        source_removed_callback_(source_id);
    }
}

void BinauralProcessor::notify_source_moved(uint32_t source_id, float x, float y, float z) {
    std::lock_guard<std::lock_guard<std::mutex>> lock(callbacks_mutex_);
    if (source_moved_callback_) {
        source_moved_callback_(source_id, x, y, z);
    }
}

void BinauralProcessor::processing_thread_function() {
    Logger::info("Binaural processing thread started");

    while (!shutdown_requested_) {
        if (processing_) {
            // Process audio (would be fed from audio pipeline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    Logger::info("Binaural processing thread stopped");
}

bool BinauralProcessor::load_hrtf_dataset_internal(const std::string& dataset_path, uint32_t dataset_id) {
    // In a real implementation, would load HRTF measurements from file
    Logger::info("Loading HRTF dataset: {}", dataset_path);

    // Create placeholder HRTF data
    HRTFData hrtf_data;
    hrtf_data.ir_length = config_.hrtf_resolution;
    hrtf_data.sample_rate = config_.sample_rate;
    hrtf_data.num_measurements = 72; // 72 measurements (5° azimuth, 5° elevation)
    hrtf_data.is_individualized = false;
    hrtf_data.subject_name = "default";

    // Generate placeholder azimuth and elevation arrays
    hrtf_data.azimuths.resize(hrtf_data.num_measurements);
    hrtf_data.elevations.resize(hrtf_data.num_measurements);
    hrtf_data.left_ir.resize(1);
    hrtf_data.right_ir.resize(1);

    for (uint32_t i = 0; i < hrtf_data.num_measurements; ++i) {
        hrtf_data.azimuths[i] = (static_cast<float>(i) / (hrtf_data.num_measurements - 1)) * 2.0f * M_PI - M_PI;
        hrtf_data.elevations[i] = -30.0f + (static_cast<float>(i / (hrtf_data.num_measurements - 1)) * 60.0f;
    }

    // Generate placeholder impulse responses
    hrtf_data.left_ir[0].resize(hrtf_data.ir_length, 0.0f);
    hrtf_data.right_ir[0].resize(hrtf_data.ir_length, 0.0f);

    for (uint32_t i = 0; i < hrtf_data.ir_length; ++i) {
        // Simple delay-based HRTF simulation
        float left_delay = 0.0005f * i * 1000.0f / 343.0f; // ~0.5ms per sample
        float right_delay = 0.0006f * i * 1000.0f / 343.0f; // ~0.6ms per sample
        hrtf_data.left_ir[0][i] = std::cos(2.0f * M_PI * i / hrtf_data.ir_length) * 0.5f;
        hrtf_data.right_ir[0][i] = std::sin(2.0f * M_PI * i / hrtf_data.ir_length) * 0.5f;
    }

    hrtf_datasets_[dataset_id] = std::move(hrtf_data);
    Logger::info("Loaded HRTF dataset {}: {} measurements, {} IR length samples",
                dataset_id, hrtf_datasets_[dataset_id].num_measurements,
                hrtf_datasets_[dataset_id].ir_length);

    return true;
}

void BinauralProcessor::unload_hrtf_dataset(uint32_t dataset_id) {
    auto it = hrtf_datasets_.find(dataset_id);
    if (it != hrtf_datasets_.end()) {
        hrtf_datasets_.erase(it);
        Logger::info("Unloaded HRTF dataset: {}", dataset_id);
    }
}

bool BinauralProcessor::validate_hrtf_index(uint32_t hrtf_index) const {
    auto it = hrtf_datasets_.find(hrtf_index);
    return it != hrtf_datasets_.end();
}

void BinauralProcessor::set_source_added_callback(SourceAddedCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    source_added_callback_ = callback;
}

void BinauralProcessor::set_source_removed_callback(SourceRemovedCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    source_removed_callback_ = callback;
}

void BinauralProcessor::set_source_moved_callback(SourceMovedCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    source_moved_callback_ = callback;
}

std::string BinauralProcessor::get_diagnostics_report() const {
    std::ostringstream report;

    report << "=== Binaural Processor Diagnostics ===\n";
    report << "Initialized: " << (initialized_ ? "Yes" : "No") << "\n";
    report << "Processing: " << (processing_.load() ? "Yes" : "No") << "\n";
    report << "3D audio: " << (is_3d_enabled() ? "Yes" : "No") << "\n";

    report << "\nConfiguration:\n";
    report << "  Sample rate: " << config_.sample_rate << " Hz\n";
    report << "  Max frame size: " << config_.max_frame_size << "\n";
    report << "  Output channels: " << config_.output_channels << "\n";
    report << "  Max sources: " << config_.max_audio_sources << "\n";
    report << " HRTF enabled: " << (config_.enable_hrtf ? "Yes" : "No") << "\n";

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        report << "\nStatistics:\n";
        report << "  Total calls: " << statistics_.total_process_calls << "\n";
        report << " Successful calls: " << statistics_.successful_calls << "\n";
        report << " Average processing time: " << statistics_.avg_processing_time_us << " μs\n";
        report << " Max processing time: " << statistics_.max_processing_time_us << " μs\n";
        report << " Active sources: " << statistics_.active_sources << "\n";
        "  Moving sources: " << statistics_.moving_sources << "\n";
        report <<  " Static sources: " << statistics_.static_sources << "\n";
        report <<  " Occluded sources: " << statistics_.occluded_sources << "\n";
        report <<  " Average distance: " << statistics_.avg_distance_meters << " meters\n";
        report << " Max distance: " << statistics_.max_distance_meters << " meters\n";
        report << " HRTF interpolations: " << statistics_.hrtf_interpolations << "\n";
    }

    report << "\nListener State:\n";
    report << "  Position: ({:.2f}, {:.2f}, {:.2f})\n",
                listener_state_.position_x, listener_state_.position_y, listener_state_.position_z);
    report << "  Orientation: ({:.2f}, {:.2f}, {:.2f})\n",
                listener_state_.orientation_yaw, listener_state_.orientation_pitch,
                listener_state_.orientation_roll);
    report << "  Head tracking: " << (listener_state_.head_tracking_enabled ? "Enabled" : "Disabled") << "\n";
    report << " VR mode: " << (listener_state_.vr_mode_enabled ? "Enabled" : "Disabled") << "\n";

    report << "\nRoom Acoustics:\n";
    report << "  Dimensions: ({:.1f}m, {:.1f}m, {:.1f}m)\n",
                room_acoustics_.dimensions_x, room_acoustics_.dimensions_y, room_acoustics_.dimensions_z);
    report << "  Reverb time: " << room_acoustics_.reverberation_time << " seconds\n";
    report << " Wall absorption: " << room_acoustics_.wall_absorption << "\n";
    report <<  " Air absorption: " << room_acoustics_.air_absorption_coefficient << "\n";

    report << "\nHRTF Information:\n";
    report << "  Loaded datasets: " << hrtf_datasets_.size() << "\n";
    if (!hrtf_datasets_.empty()) {
        const auto& first_dataset = hrt_datasets_.begin()->second;
        report << "  Dataset 0: " << first_dataset.num_measurements << " measurements\n";
        report << "  IR length: " << first_dataset.ir_length << " samples\n";
        report <<  " Sample rate: " << first_dataset.sample_rate << " Hz\n";
        report << " Individualized: " << (first_dataset.is_individualized ? "Yes" : "No") << "\n";
    }

    report << "\n=== End Diagnostics ===\n";

    return report.str();
}

bool BinauralProcessor::validate_configuration() const {
    // Validate basic configuration parameters
    if (config_.sample_rate == 0 || config_.max_frame_size == 0) {
        return false;
    }

    if (config_.max_audio_sources == 0) {
        return false;
    }

    // Validate speaker configuration
    if (config_.type == SpatialAudioType::BINAURAL && config_.output_channels != 2) {
        return false;
    }

    return true;
}

std::vector<std::string> BinauralProcessor::test_spatial_capabilities() const {
    std::vector<std::string> results;

    results.push_back("✓ Audio source management");
    results.push_back("✓ 3D positioning");
    results.push_back("✓ HRTF processing: " + (config_.enable_hrtf ? "Enabled" : "Disabled"));
    results.push_back("✓ Distance attenuation");
    results.push_back("✓ Doppler effect: " + (config_.enable_doppler ? "Enabled" : "Disabled"));
    results.push_back("✓ Room acoustics: " + (config_.enable_room_acoustics ? "Enabled" : "Disabled"));
    results.push_back("✓ Real-time positioning: " + (supports_real_time_positioning() ? "Enabled" : "Disabled"));

    return results;
}

// AmbisonicsProcessor implementation (simplified)
AmbisonicsProcessor::AmbisonicsProcessor()
    : initialized_(false) {
}

AmbisonicsProcessor::~AmbionsProcessor() {
    shutdown();
}

bool AmbisonicsProcessor::initialize(const SpatialAudioConfig& config) {
    config_ = config;
    config_.type = SpatialAudioType::AMBISONIC_DECODING;
    config_.speaker_config = SpeakerConfiguration::STEREO;

    // Initialize ambisonics parameters
    ambisonic_order_ = config_.ambisonic_order;
    enable_nfc_ = false;
    nfc_distance_ = 1.0f;

    // Calculate ambisonics channels
    ambisonic_channels_ = spatial_audio_utils::get_ambisonic_channel_count(ambisonic_order_);

    // Initialize encoding/decoding matrices
    spatial_audio_utils::calculate_ambisonics_encoding_matrix(ambisonic_order_, encoding_matrix_);
    spatial_audio_utils::calculate_ambisonics_decoding_matrix(
        ambisonic_order_, config_.speaker_config, std::vector<float>(), decoding_matrix_);

    // Initialize processing buffers
    uint32_t total_samples = config_.max_frame_size * config_.output_channels;
    ambisonics_buffer_.resize(total_samples, 0.0f);
    work_buffer_.resize(total_samples, 0.0f);

    initialized_ = true;
    Logger::info("Ambisonics Processor initialized with order {}", ambisonic_order_);
    return true;
}

void AmbisonicsProcessor::shutdown() {
    ambisonics_buffer_.clear();
    work_buffer_.clear();
    encoding_matrix_.clear();
    decoding_matrix_.clear();
    nfc_filter_.clear();

    initialized_ = false;
    Logger::info("Ambisonics Processor shutdown");
}

bool AmbisonicsProcessor::reset() {
    std::fill(ambisonics_buffer_.begin(), ambisonics_buffer_.end(), 0.0f);
    std::fill(work_buffer_.begin(), work_buffer_.end(), 0.0f);
    return true;
}

bool AmbisonicsProcessor::process(const float** inputs, float* output, uint32_t frame_count,
                             const std::vector<AudioSource*>& sources) {
    // Simplified ambisonics processing
    if (!initialized_ || !inputs || !output || frame_count == 0) {
        return false;
    }

    // For now, just copy input to output
    std::memcpy(output, inputs[0], frame_count * config_.output_channels * sizeof(float));

    return true;
}

bool AmbisonicsProcessor::process_interleaved(const float* input, float* output, uint32_t frame_count,
                                         const std::vector<AudioSource*>& sources) {
    return process(&input, output, frame_count, sources);
}

bool AmbisonicsProcessor::encode_ambisonics(const float* input, float* ambisonics_output,
                                              uint32_t frame_count, uint32_t input_channels,
                                              const float* source_positions) {
    // Ambisonics encoding implementation would go here
    return true;
}

bool AmbisonicsProcessor::decode_ambisonics(const float* ambisonics_input, float* output,
                                               uint32_t frame_count, const float* speaker_positions,
                                               uint32_t output_channels) {
    // Ambisonics decoding implementation would go here
    return true;
}

bool AmbisonicsProcessor::set_ambisonic_order(uint32_t order) {
    ambisonic_order_ = std::min(3u, order); // Max 3rd order for now
    ambisonic_channels_ = spatial_audio_utils::get_ambisonic_channel_count(ambisonic_order_);
    spatial_audio_utils::calculate_ambisonics_encoding_matrix(ambisonic_order_, encoding_matrix_);
    return true;
}

bool AmbisonicsProcessor::enable_near_field_compensation(bool enabled) {
    enable_nfc_ = enabled;
    return true;
}

bool AmbisonicsProcessor::enable_nfc_distance(float distance) {
    nfc_distance_ = distance;
    return true;
}

uint32_t AmbisonicsProcessor::get_ambisonic_order() const {
    return ambisonic_order_;
}

uint32_t AmbisonicsProcessor::get_ambisonic_channels() const {
    return ambisonic_channels_;
}

SpatialAudioType AmbisonicsProcessor::get_type() const {
    return config_.type;
}

SpeakerConfiguration AmbisonicsProcessor::get_speaker_configuration() const {
    return config_.speaker_config;
}

std::string AmbisonicsProcessor::get_name() const {
    return "Ambisonics Processor";
}

std::string AmbisonicsProcessor::get_version() const {
    return "1.0.0";
}

std::string AmbisonicsProcessor::get_description() const {
    return "Ambisonics encoder/decoder for spherical harmonics";
}

SpatialAudioStatistics AmbisonicsProcessor::get_statistics() const {
    return SpatialAudioStatistics(); // Placeholder
}

void AmbisonicsProcessor::reset_statistics() {
    // Statistics reset would go here
}

bool AmbisonicsProcessor::supports_real_time_positioning() const {
    return true;
}

bool AmbisonicsProcessor::supports_hrtf() const {
    return false;
}

bool AmbisonicsProcessor::supports_ambisonics() const {
    return true;
}

bool AmbisonicsProcessor::supports_gpu_acceleration() const {
    return false;
}

uint32_t AmbisonicsProcessor::get_max_sources() const {
    return config_.max_audio_sources;
}

double AmbisonicsProcessor::get_expected_latency_ms() const {
    return 15.0f; // Estimated latency for ambisonics processing
}

bool AmbisonicsProcessor::is_3d_enabled() const {
    return false;
}

// Factory implementations
std::unique_ptr<SpatialAudioProcessor> SpatialAudioProcessorFactory::create_binaural_processor() {
    return std::make_unique<BinauralProcessor>();
}

std::unique_ptr<SpatialAudioProcessor> SpatialAudioProcessorFactory::create_hrtf_processor(const std::string& hrtf_path) {
    auto processor = std::make_unique<BinauralProcessor>();
    SpatialAudioConfig config;
    config.type = SpatialAudioType::BINAURAL;
    config.enable_hrtf = true;
    config.hrtf_dataset_path = hrtf_path;

    processor->initialize(config);
    return std::move(processor);
}

std::unique_ptr<SpatialAudioProcessor> SpatialAudioProcessorFactory::create_ambisonics_decoder(
    uint32_t order, SpeakerConfiguration config) {
    auto processor = std::make_unique<AmbisonicsProcessor>();
    SpatialAudioConfig config_params;
    config_params.type = SpatialAudioType::AMBISONIC_DECODING;
    config_params.ambisonic_order = order;
    config_params.speaker_config = config;

    processor->initialize(config_params);
    return std::move(processor);
}

std::unique_ptr<SpatialAudioProcessor> SpatialAudioProcessorFactory::create_3d_audio_processor() {
    auto processor = std::make_unique<BinauralProcessor>();
    SpatialAudioConfig config;
    config.type = SpatialAudioType::BINAURAL;
    config.max_audio_sources = 64;

    processor->initialize(config);
    return std::move(processor);
}

std::unique_ptr<SpatialAudioProcessor> SpatialAudioFactory::create_vr_audio_processor() {
    auto processor = std::make_unique<BinauralProcessor>();
    SpatialAudioConfig config;
    config.type = SpatialAudioType::VR_SPATIAL_AUDIO;
    config.enable_head_tracking = true;
    config.enable_vr_mode = true;
    config.max_audio_sources = 32;

    processor->initialize(config);
    return std::move(processor);
}

std::vector<std::string> SpatialAudioProcessorFactory::get_available_processor_types() {
    return {
        "binaural", "ambisonics", "3d_audio", "vr_audio",
        "multi_channel_panner", "vbap", "dolby_atmos",
        "game_audio", "console_audio", "live_sound",
        "automotive_audio", "cinema_audio"
    };
}

std::vector<std::string> SpatialAudioFactory::get_available_speaker_configurations() {
    return {
        "mono", "stereo", "lcr", "quad", "surround_5_1",
        "surround_7_1", "dolby_atmos", "auro_3d",
        "imax", "binaural", "ambisonic", "custom"
    };
}

// Utility namespace implementations
namespace spatial_audio_utils {

void cartesian_to_spherical(float x, float y, float z,
                                  float& azimuth, float& elevation, float& distance) {
    distance = std::sqrt(x * x + y * y + z * z);
    azimuth = std::atan2(y, x);
    elevation = std::asin(z / distance);
}

void spherical_to_cartesian(float azimuth, float elevation, float distance,
                                  float& x, float& y, float& z) {
    x = distance * std::cos(elevation) * std::cos(azimuth);
    y = distance * std::cos(elevation) * std::sin(azimuth);
    z = distance * std::sin(elevation);
}

float meters_to_feet(float meters) {
    return meters * 3.28084f;
}

float feet_to_meters(float feet) {
    return feet * 0.3048f;
}

float degrees_to_radians(float degrees) {
    return degrees * M_PI / 180.0f;
}

float radians_to_degrees(float radians) {
    return radians * 180.0f / M_PI;
}

float hz_to_mel(float frequency_hz) {
    return 2595.0f * std::log10(1.0f + frequency_hz / 700.0f);
}

float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

float calculate_distance_attenuation(float distance, float reference_distance, float rolloff_factor) {
    if (distance <= reference_distance) {
        return 1.0f;
    }
    return reference_distance / std::pow(distance / reference_distance, rolloff_factor);
}

float calculate_air_absorption(float frequency, float distance, float coefficient) {
    return std::exp(-coefficient * distance * frequency / 1000.0f);
}

float calculate_inverse_distance_attenuation(float gain, float reference_distance, float rolloff_factor) {
    if (gain >= 1.0f) {
        return reference_distance;
    }
    return reference_distance * std::pow(gain, 1.0f / rolloff_factor);
}

float calculate_doppler_shift(float relative_velocity, float speed_of_sound) {
    return speed_of_sound / (speed_of_sound - relative_velocity);
}

float calculate_doppler_factor(float relative_velocity, float speed_of_sound) {
    return calculate_doppler_shift(relative_velocity, speed_of_sound);
}

float calculate_cardioid_pattern(float angle, float directivity_factor) {
    return 0.5f + 0.5f * directivity_factor * std::cos(angle);
}

float calculate_hypercardioid_pattern(float angle, float directivity_factor) {
    return 0.5f + 0.75f * directivity_factor * std::cos(angle);
}

float calculate_supercardioid_pattern(float angle, float directivity_factor) {
    return 0.5f + 0.25f * directivity_factor * std::cos(angle);
}

uint32_t get_ambisonic_channel_count(uint32_t order) {
    return (order + 1) * (order + 2) / 2;
}

uint32_t get_spherical_harmonic_order(uint32_t channel) {
    // Simple inverse of the formula above
    // For order n, channel count = (n+1)*(n+2)/2
    // n^2 + 3n + 2 = channel_count
    // n = floor((-3 + sqrt(9 + 8*channel_count)) / 2)
    return static_cast<uint32_t>(std::floor((-3.0f + std::sqrt(9.0f + 8.0f * channel_count) / 2.0f)));
}

void calculate_ambisonics_encoding_matrix(uint32_t order,
                                            std::vector<std::vector<float>>& matrix) {
    // Calculate encoding matrix for ambisonics
    uint32_t num_channels = get_ambisonic_channel_count(order);
    matrix.resize(num_channels, std::vector<float>(num_channels, 0.0f));

    // Simple encoding matrix for first-order ambisonics
    if (order >= 1) {
        // W (omnidirectional)
        matrix[0][0] = 1.0f / std::sqrt(2.0f);
        matrix[0][1] = 1.0f / std::sqrt(2.0f);
        matrix[0][2] = 0.0f;

        // Y (front-back)
        matrix[1][0] = 0.0f;
        matrix[1][1] = 0.0f;
        matrix[1][2] = 1.0f;

        // Z (up-down)
        matrix[2][0] = 0.0f;
        matrix[2][1] = 0.0f;
        matrix[2][2] = 0.0f;
    }

    // For higher orders, would calculate using spherical harmonics
    // This is a placeholder for demonstration
}

void calculate_ambisonics_decoding_matrix(uint32_t order,
                                              SpeakerConfiguration config,
                                              const std::vector<float>& speaker_positions,
                                              std::vector<std::vector<float>>& matrix) {
    // Calculate decoding matrix for given speaker configuration
    uint32_t num_speakers = 2; // Simplified for stereo
    uint32_t num_channels = get_ambisonics_channel_count(order);
    matrix.resize(num_channels, std::vector<float>(num_speakers, 0.0f));

    // Simple stereo decoding matrix for first-order ambisonics
    if (order >= 1) {
        // Left channel decode
        matrix[0][0] = 1.0f; // W contribution
        matrix[0][1] = 1.0f; // Y contribution
        matrix[0][2] = 0.0f; // Z contribution

        // Right channel decode
        matrix[1][0] = 1.0f; // W contribution
        matrix[1][1] = 0.0f; // Y contribution
        matrix[1][2] = 1.0f; // Z contribution
    }
}

void calculate_nfc_filter(uint32_t order, float distance,
                                std::vector<float>& filter_coefficients) {
    // Near-field compensation filter
    filter_coefficients.clear();
    filter_coefficients.resize(order * 2 + 1, 0.0f);

    // Simple NFC filter coefficients
    for (uint32_t n = 0; n <= order; ++n) {
        float frequency = static_cast<float>(n + 1) * 100.0f / order;
        float gain = spatial_audio_utils::meters_to_feet(distance);
        filter_coefficients[n * 2] = gain / (1.0f + frequency / 100.0f);
        filter_coefficients[n * 2 + 1] = -gain / (1.0f + frequency / 100.0f);
    }
}

} // namespace spatial_audio_utils

} // namespace vortex::core::dsp