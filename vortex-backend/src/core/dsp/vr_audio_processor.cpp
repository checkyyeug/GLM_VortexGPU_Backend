#include "core/dsp/vr_audio_processor.hpp"
#include "core/dsp/audio_buffer.hpp"
#include "core/dsp/audio_math.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>
#include <random>

#ifdef VORTEX_ENABLE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#endif

#ifdef VORTEX_ENABLE_OPENCL
#include <CL/cl.h>
#endif

namespace vortex {
namespace core {
namespace dsp {

// VR Audio constants
constexpr float DEFAULT_SOUND_SPEED = 343.0f; // m/s at 20°C
constexpr float MIN_DISTANCE = 0.01f; // Minimum distance to avoid division by zero
constexpr float MAX_DISTANCE = 1000.0f; // Maximum effective distance
constexpr float MIN_OCCLUSION = 0.0f; // Minimum occlusion factor
constexpr float MAX_OCCLUSION = 1.0f; // Maximum occlusion factor
constexpr float QUALITY_ADAPTATION_THRESHOLD = 0.8f; // CPU usage threshold for quality adaptation
constexpr float QUALITY_ADAPTATION_HYSTERESIS = 0.1f; // Hysteresis for quality adaptation
constexpr size_t HRTF_CACHE_MAX_SIZE = 10000; // Maximum HRTF cache entries
constexpr int HRTF_INTERPOLATION_NEIGHBORS = 4; // Number of neighbors for HRTF interpolation

VRAudioProcessor::VRAudioProcessor()
    : last_metrics_update_(std::chrono::high_resolution_clock::now()),
      last_quality_adaptation_(std::chrono::high_resolution_clock::now()) {

    // Initialize core components
    spatial_processor_ = std::make_unique<SpatialAudioProcessor>();
    spectrum_analyzer_ = std::make_unique<SpectrumAnalyzer>();
    waveform_processor_ = std::make_unique<WaveformProcessor>();
    vu_processor_ = std::make_unique<VUMeterProcessor>();

    // Initialize pose
    std::memset(&current_head_pose_, 0, sizeof(current_head_pose_));
    current_head_pose_.orientation[3] = 1.0f; // Identity quaternion
    current_head_pose_.is_valid = true;
    current_head_pose_.confidence = 1.0f;

    // Initialize room acoustics with defaults
    room_acoustics_.room_dimensions[0] = 10.0f; // width
    room_acoustics_.room_dimensions[1] = 8.0f;  // height
    room_acoustics_.room_dimensions[2] = 3.0f;  // depth
    room_acoustics_.reflection_coefficient = 0.5f;
    room_acoustics_.reverb_decay_time = 1.5f;
    room_acoustics_.diffusion = 0.8f;

    // Initialize quality settings
    quality_settings_.hrtf_quality = VRRenderingQuality::HIGH;
    quality_settings_.reverb_quality = VRRenderingQuality::MEDIUM;
    quality_settings_.occlusion_quality = VRRenderingQuality::MEDIUM;
    quality_settings_.reflection_quality = VRRenderingQuality::LOW;
    quality_settings_.enable_gpu_acceleration = true;
    quality_settings_.enable_multi_threading = true;
    quality_settings_.max_concurrent_sources = 32;
    quality_settings_.processing_latency_target = 10.0f;
    quality_settings_.spatial_sampling_rate = 48000;
    quality_settings_.hrtf_impulse_response_length = 256;
}

VRAudioProcessor::~VRAudioProcessor() {
    shutdown();
}

bool VRAudioProcessor::initialize(int sample_rate, int buffer_size, int channels,
                                 const VRRenderingQuality& quality) {
    if (initialized_) {
        return false;
    }

    sample_rate_ = sample_rate;
    buffer_size_ = buffer_size;
    channels_ = channels;
    quality_settings_ = quality;

    // Initialize core components
    if (!spatial_processor_->initialize(sample_rate, buffer_size, channels)) {
        return false;
    }

    if (!spectrum_analyzer_->initialize(sample_rate, buffer_size, 2048,
                                        SpectrumAnalyzer::WindowType::Hanning)) {
        return false;
    }

    if (!waveform_processor_->initialize(sample_rate, buffer_size)) {
        return false;
    }

    if (!vu_processor_->initialize(sample_rate, buffer_size, channels)) {
        return false;
    }

    // Initialize GPU acceleration if requested and available
    if (quality_settings_.enable_gpu_acceleration) {
        initializeGPU();
    }

    // Load HRTF data
    if (!loadHRTFData()) {
        // Use built-in HRTF data if loading fails
        hrtf_data_path_ = "";
    }

    // Clear pose history
    pose_history_.clear();

    // Initialize metrics
    std::memset(&metrics_, 0, sizeof(metrics_));
    last_metrics_update_ = std::chrono::high_resolution_clock::now();
    total_frames_processed_ = 0;

    initialized_ = true;
    return true;
}

void VRAudioProcessor::shutdown() {
    if (!initialized_) {
        return;
    }

    // Stop all processing
    initialized_ = false;

    // Clear audio sources
    {
        std::lock_guard<std::mutex> lock(sources_mutex_);
        audio_sources_.clear();
    }

    // Shutdown core components
    if (spatial_processor_) {
        spatial_processor_->shutdown();
    }

    // Shutdown GPU
    shutdownGPU();

    // Clear caches
    {
        std::lock_guard<std::mutex> lock(hrtf_mutex_);
        hrtf_cache_.clear();
    }

    // Clear callbacks
    metrics_callback_ = nullptr;
    head_pose_callback_ = nullptr;
}

bool VRAudioProcessor::processAudioFrame(const float* input_buffer, float* output_buffer,
                                        size_t num_samples, const HeadPose& current_head_pose) {
    if (!initialized_ || !input_buffer || !output_buffer || num_samples == 0) {
        return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Update head pose
    updateHeadPose(current_head_pose);

    // Clear output buffer
    std::memset(output_buffer, 0, num_samples * channels_ * sizeof(float));

    // Process audio sources
    processAudioSources(output_buffer, num_samples);

    // Apply room acoustics (reverb, early reflections)
    if (room_acoustics_.enable_late_reverb || room_acoustics_.enable_early_reflections) {
        applyReverb(output_buffer, num_samples);
    }

    // Update audio processors
    spectrum_analyzer_->processAudio(input_buffer, num_samples);
    waveform_processor_->processAudio(input_buffer, num_samples);
    vu_processor_->processAudio(input_buffer, num_samples);

    // Update metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    double processing_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.processing_time_ms = processing_time;
        total_frames_processed_++;

        if (total_frames_processed_ % 60 == 0) { // Update every second at 60fps
            updateMetrics();
        }
    }

    // Quality adaptation if enabled
    if (quality_adaptation_enabled_) {
        adaptQuality();
    }

    return true;
}

uint32_t VRAudioProcessor::addAudioSource(const VRAudioSource& source) {
    if (!initialized_ || !validateSource(source)) {
        return 0;
    }

    uint32_t source_id = next_source_id_++;

    auto source_state = std::make_unique<VRAudioSourceState>();
    source_state->config = source;
    source_state->audio_buffer.resize(buffer_size_ * source_id, 0.0f);
    source_state->buffer_position = 0;
    source_state->last_distance = vr_audio_utils::calculateDistance(source.position, current_head_pose_.position);
    source_state->is_active = true;
    source_state->current_gain = source.gain;
    source_state->target_gain = source.gain;
    source_state->last_update_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    // Preload HRTF data for this source position
    auto hrtf_irs = getHRTFImpulseResponse(source.position, current_head_pose_.position,
                                           current_head_pose_.orientation);
    source_state->hrtf_ir_left = hrtf_irs.first;
    source_state->hrtf_ir_right = hrtf_irs.second;

    {
        std::lock_guard<std::mutex> lock(sources_mutex_);
        audio_sources_[source_id] = std::move(source_state);
    }

    return source_id;
}

bool VRAudioProcessor::removeAudioSource(uint32_t source_id) {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = audio_sources_.find(source_id);
    if (it == audio_sources_.end()) {
        return false;
    }

    audio_sources_.erase(it);
    return true;
}

bool VRAudioProcessor::updateAudioSource(uint32_t source_id, const VRAudioSource& source) {
    if (!validateSource(source)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = audio_sources_.find(source_id);
    if (it == audio_sources_.end()) {
        return false;
    }

    it->second->config = source;
    it->second->target_gain = source.gain;
    it->second->last_update_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    // Update HRTF data if position changed
    float old_distance = it->second->last_distance;
    float new_distance = vr_audio_utils::calculateDistance(source.position, current_head_pose_.position);

    if (std::abs(new_distance - old_distance) > 0.1f) { // 10cm threshold
        auto hrtf_irs = getHRTFImpulseResponse(source.position, current_head_pose_.position,
                                               current_head_pose_.orientation);
        it->second->hrtf_ir_left = hrtf_irs.first;
        it->second->hrtf_ir_right = hrtf_irs.second;
        it->second->last_distance = new_distance;
    }

    return true;
}

std::optional<VRAudioSource> VRAudioProcessor::getAudioSource(uint32_t source_id) const {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = audio_sources_.find(source_id);
    if (it == audio_sources_.end()) {
        return std::nullopt;
    }

    return it->second->config;
}

void VRAudioProcessor::setRoomAcoustics(const VRRoomAcoustics& acoustics) {
    room_acoustics_ = acoustics;
}

VRRoomAcoustics VRAudioProcessor::getRoomAcoustics() const {
    return room_acoustics_;
}

void VRAudioProcessor::setRenderingMode(VRRenderingMode mode) {
    rendering_mode_ = mode;
}

VRRenderingMode VRAudioProcessor::getRenderingMode() const {
    return rendering_mode_;
}

void VRAudioProcessor::setHeadTrackingMode(HeadTrackingMode mode) {
    head_tracking_mode_ = mode;
}

HeadTrackingMode VRAudioProcessor::getHeadTrackingMode() const {
    return head_tracking_mode_;
}

bool VRAudioProcessor::configureHRTF(const std::string& hrtf_path, bool enable_individualized,
                                    float ear_distance) {
    hrtf_data_path_ = hrtf_path;
    hrtf_individualized_ = enable_individualized;
    ear_distance_ = ear_distance;

    // Clear existing cache
    {
        std::lock_guard<std::mutex> lock(hrtf_mutex_);
        hrtf_cache_.clear();
    }

    return loadHRTFData();
}

void VRAudioProcessor::setDopplerEnabled(bool enable) {
    doppler_enabled_ = enable;
}

void VRAudioProcessor::setOcclusionEnabled(bool enable, OcclusionMethod method) {
    occlusion_enabled_ = enable;
    occlusion_method_ = method;
}

void VRAudioProcessor::setReflectionEnabled(bool enable, int max_order) {
    reflection_enabled_ = enable;
    room_acoustics_.max_reflection_order = max_order;
}

void VRAudioProcessor::setQualityAdaptation(bool enable, float target_latency,
                                           VRRenderingQuality::Level min_quality) {
    quality_adaptation_enabled_ = enable;
    target_latency_ms_ = target_latency;
    min_quality_level_ = min_quality;
    current_quality_level_ = static_cast<int>(quality_settings_.hrtf_quality);
}

void VRAudioProcessor::updateHeadPose(const HeadPose& pose) {
    if (!pose.is_valid || pose.confidence < 0.1f) {
        return;
    }

    std::lock_guard<std::mutex> lock(pose_mutex_);

    // Add to history
    pose_history_.push_back({pose, std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count()});

    // Limit history size
    while (pose_history_.size() > POSE_HISTORY_SIZE) {
        pose_history_.erase(pose_history_.begin());
    }

    last_head_pose_ = current_head_pose_;
    current_head_pose_ = pose;

    // Notify callback if set
    if (head_pose_callback_) {
        head_pose_callback_(pose);
    }
}

HeadPose VRAudioProcessor::predictHeadPose(float future_time) const {
    std::lock_guard<std::mutex> lock(pose_mutex_);

    if (pose_history_.size() < 2 || future_time <= 0.0f) {
        return current_head_pose_;
    }

    // Simple linear prediction based on recent velocity
    HeadPose predicted = current_head_pose_;

    // Update position
    predicted.position[0] += current_head_pose_.velocity[0] * future_time;
    predicted.position[1] += current_head_pose_.velocity[1] * future_time;
    predicted.position[2] += current_head_pose_.velocity[2] * future_time;

    // Update orientation (simplified - use angular velocity)
    if (std::abs(current_head_pose_.angular_velocity[0]) > 0.001f ||
        std::abs(current_head_pose_.angular_velocity[1]) > 0.001f ||
        std::abs(current_head_pose_.angular_velocity[2]) > 0.001f) {

        // Convert angular velocity to quaternion change
        float angle = std::sqrt(current_head_pose_.angular_velocity[0] * current_head_pose_.angular_velocity[0] +
                               current_head_pose_.angular_velocity[1] * current_head_pose_.angular_velocity[1] +
                               current_head_pose_.angular_velocity[2] * current_head_pose_.angular_velocity[2]) * future_time;

        if (angle > 0.001f) {
            float axis_x = current_head_pose_.angular_velocity[0] / angle;
            float axis_y = current_head_pose_.angular_velocity[1] / angle;
            float axis_z = current_head_pose_.angular_velocity[2] / angle;

            float half_angle = angle * 0.5f;
            float sin_half = std::sin(half_angle);
            float cos_half = std::cos(half_angle);

            float rotation_quat[4] = {
                axis_x * sin_half,
                axis_y * sin_half,
                axis_z * sin_half,
                cos_half
            };

            // Multiply current orientation by rotation
            vr_audio_utils::multiplyQuaternions(current_head_pose_.orientation, rotation_quat,
                                              predicted.orientation);
            vr_audio_utils::normalizeQuaternion(predicted.orientation);
        }
    }

    predicted.timestamp += static_cast<uint64_t>(future_time * 1000000000.0); // Convert to nanoseconds
    predicted.prediction_time = future_time;

    return predicted;
}

HeadPose VRAudioProcessor::getCurrentHeadPose() const {
    std::lock_guard<std::mutex> lock(pose_mutex_);
    return current_head_pose_;
}

bool VRAudioProcessor::setAudioSourceData(uint32_t source_id, const float* audio_data, size_t num_samples) {
    if (!audio_data || num_samples == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = audio_sources_.find(source_id);
    if (it == audio_sources_.end()) {
        return false;
    }

    // Resize buffer if needed
    if (it->second->audio_buffer.size() < num_samples) {
        it->second->audio_buffer.resize(num_samples, 0.0f);
    }

    // Copy audio data
    std::copy(audio_data, audio_data + num_samples, it->second->audio_buffer.begin());
    it->second->buffer_position = 0;
    it->second->is_active = true;

    return true;
}

bool VRAudioProcessor::streamAudioToSource(uint32_t source_id, const float* audio_data, size_t num_samples) {
    if (!audio_data || num_samples == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(sources_mutex_);

    auto it = audio_sources_.find(source_id);
    if (it == audio_sources_.end()) {
        return false;
    }

    // Add to queue
    {
        std::lock_guard<std::mutex> queue_lock(it->second->queue_mutex);
        for (size_t i = 0; i < num_samples; ++i) {
            it->second->audio_queue.push(audio_data[i]);
        }

        // Limit queue size to prevent memory issues
        while (it->second->audio_queue.size() > buffer_size_ * 4) {
            it->second->audio_queue.pop();
        }
    }

    it->second->is_active = true;
    return true;
}

void VRAudioProcessor::setListenerPose(const float position[3], const float orientation[4]) {
    HeadPose pose;
    pose.position[0] = position[0];
    pose.position[1] = position[1];
    pose.position[2] = position[2];
    pose.orientation[0] = orientation[0];
    pose.orientation[1] = orientation[1];
    pose.orientation[2] = orientation[2];
    pose.orientation[3] = orientation[3];
    pose.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    pose.is_valid = true;
    pose.confidence = 1.0f;

    updateHeadPose(pose);
}

HeadPose VRAudioProcessor::getListenerPose() const {
    return getCurrentHeadPose();
}

void VRAudioProcessor::setRenderingQuality(const VRRenderingQuality& quality) {
    quality_settings_ = quality;

    // Update component settings based on quality
    // This would require additional interface methods in the component classes
}

VRRenderingQuality VRAudioProcessor::getRenderingQuality() const {
    return quality_settings_;
}

VRAudioMetrics VRAudioProcessor::getMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void VRAudioProcessor::setMetricsCallback(VRAudioMetricsCallback callback) {
    metrics_callback_ = callback;
}

void VRAudioProcessor::setHeadPoseCallback(HeadPoseCallback callback) {
    head_pose_callback_ = callback;
}

bool VRAudioProcessor::preloadHRTFPositions(const std::vector<std::array<float, 3>>& positions) {
    for (const auto& pos : positions) {
        auto hrtf_irs = getHRTFImpulseResponse(pos.data(), current_head_pose_.position,
                                               current_head_pose_.orientation);
        if (hrtf_irs.first.empty() || hrtf_irs.second.empty()) {
            return false;
        }
    }
    return true;
}

void VRAudioProcessor::clearHRTFCache() {
    std::lock_guard<std::mutex> lock(hrtf_mutex_);
    hrtf_cache_.clear();
}

std::pair<float, size_t> VRAudioProcessor::getHRTFCacheStats() const {
    std::lock_guard<std::mutex> lock(hrtf_mutex_);

    size_t total_requests = metrics_.hrtf_cache_hits + metrics_.hrtf_cache_misses;
    float hit_ratio = total_requests > 0 ?
                     static_cast<float>(metrics_.hrtf_cache_hits) / total_requests : 0.0f;

    return {hit_ratio, hrtf_cache_.size()};
}

float VRAudioProcessor::performOcclusionRaycast(const float from_pos[3], const float to_pos[3]) {
    if (occlusion_method_ == OcclusionMethod::NONE) {
        return 1.0f; // No occlusion
    }

    // Simple approximate occlusion based on room boundaries
    if (occlusion_method_ == OcclusionMethod::APPROXIMATE) {
        // Check if line passes through walls
        bool occluded = false;
        float t = 0.0f;

        while (t <= 1.0f && !occluded) {
            float check_pos[3] = {
                from_pos[0] + t * (to_pos[0] - from_pos[0]),
                from_pos[1] + t * (to_pos[1] - from_pos[1]),
                from_pos[2] + t * (to_pos[2] - from_pos[2])
            };

            // Check room boundaries
            if (check_pos[0] <= 0.0f || check_pos[0] >= room_acoustics_.room_dimensions[0] ||
                check_pos[1] <= 0.0f || check_pos[1] >= room_acoustics_.room_dimensions[1] ||
                check_pos[2] <= 0.0f || check_pos[2] >= room_acoustics_.room_dimensions[2]) {
                occluded = true;
            }

            t += 0.1f; // Check every 10% of the path
        }

        return occluded ? 0.3f : 1.0f; // 70% occlusion if blocked
    }

    // For other methods, would need more complex ray casting implementation
    return 0.8f; // Default partial occlusion
}

std::vector<std::pair<float, float>> VRAudioProcessor::calculateEarlyReflections(
    const float source_pos[3], const float listener_pos[3],
    const float room_dims[3], float reflection_coefficient) {

    std::vector<std::pair<float, float>> reflections;

    if (!reflection_enabled_ || room_acoustics_.max_reflection_order == 0) {
        return reflections;
    }

    // Calculate first-order reflections from each wall
    // This is a simplified calculation - full implementation would be more complex

    // Left wall reflection
    float left_image[3] = {-source_pos[0], source_pos[1], source_pos[2]};
    float left_distance = vr_audio_utils::calculateDistance(left_image, listener_pos);
    float left_delay = left_distance / DEFAULT_SOUND_SPEED;
    float left_gain = reflection_coefficient / (1.0f + left_distance);
    reflections.emplace_back(left_delay, left_gain);

    // Right wall reflection
    float right_image[3] = {2.0f * room_dims[0] - source_pos[0], source_pos[1], source_pos[2]};
    float right_distance = vr_audio_utils::calculateDistance(right_image, listener_pos);
    float right_delay = right_distance / DEFAULT_SOUND_SPEED;
    float right_gain = reflection_coefficient / (1.0f + right_distance);
    reflections.emplace_back(right_delay, right_gain);

    // Similar calculations for floor and ceiling...

    return reflections;
}

bool VRAudioProcessor::validateConfiguration() const {
    return initialized_ &&
           sample_rate_ > 0 &&
           buffer_size_ > 0 &&
           channels_ > 0 &&
           quality_settings_.max_concurrent_sources > 0 &&
           ear_distance_ > 0.0f;
}

void VRAudioProcessor::reset() {
    // Clear all audio sources
    {
        std::lock_guard<std::mutex> lock(sources_mutex_);
        for (auto& [id, state] : audio_sources_) {
            state->buffer_position = 0;
            state->is_active = false;
            std::fill(state->audio_buffer.begin(), state->audio_buffer.end(), 0.0f);
            while (!state->audio_queue.empty()) {
                state->audio_queue.pop();
            }
        }
    }

    // Reset pose
    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        std::memset(&current_head_pose_, 0, sizeof(current_head_pose_));
        current_head_pose_.orientation[3] = 1.0f; // Identity quaternion
        current_head_pose_.is_valid = true;
        current_head_pose_.confidence = 1.0f;
        pose_history_.clear();
    }

    // Reset metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        std::memset(&metrics_, 0, sizeof(metrics_));
        total_frames_processed_ = 0;
    }
}

bool VRAudioProcessor::initializeGPU() {
#ifdef VORTEX_ENABLE_CUDA
    // Initialize CUDA for HRTF convolution and spatial processing
    cudaError_t cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        return false;
    }

    // Create CUDA streams for parallel processing
    // ... GPU initialization code would go here

    gpu_initialized_ = true;
    return true;
#else
    return false;
#endif
}

void VRAudioProcessor::shutdownGPU() {
    if (gpu_initialized_) {
#ifdef VORTEX_ENABLE_CUDA
        // Cleanup CUDA resources
        // ... GPU cleanup code would go here
#endif
        gpu_initialized_ = false;
    }
}

bool VRAudioProcessor::loadHRTFData() {
    // This would load HRTF data from files or use built-in data
    // For now, we'll create a simple default HRTF
    return true;
}

void VRAudioProcessor::processAudioSources(float* output_buffer, size_t num_samples) {
    std::lock_guard<std::mutex> lock(sources_mutex_);

    if (audio_sources_.empty()) {
        return;
    }

    // Process each active source
    std::vector<const float*> source_buffers;

    for (const auto& [source_id, state] : audio_sources_) {
        if (!state->is_active) {
            continue;
        }

        // Process individual source
        processIndividualSource(state.get(), output_buffer, num_samples);
    }
}

void VRAudioProcessor::processIndividualSource(VRAudioSourceState* source, float* output_buffer, size_t num_samples) {
    const auto& config = source->config;

    // Calculate source-listener distance and relative position
    float distance = vr_audio_utils::calculateDistance(config.position, current_head_pose_.position);
    distance = std::max(distance, MIN_DISTANCE);

    // Apply distance attenuation
    float distance_gain = 1.0f;
    calculateDistanceAttenuation(config, distance, distance_gain);

    // Apply directivity
    float directivity_gain = 1.0f;
    if (config.directivity > 0.0f) {
        calculateDirectivityGain(config, current_head_pose_.position, directivity_gain);
    }

    // Apply occlusion
    float occlusion_factor = 1.0f;
    if (config.enable_occlusion && occlusion_enabled_) {
        source->last_occlusion = performOcclusionRaycast(config.position, current_head_pose_.position);
        occlusion_factor = source->last_occlusion;
    }

    // Calculate final gain
    float total_gain = config.gain * distance_gain * directivity_gain * occlusion_factor;

    // Smooth gain changes to avoid clicks
    const float gain_smoothing_factor = 0.95f;
    source->current_gain = source->current_gain * gain_smoothing_factor + total_gain * (1.0f - gain_smoothing_factor);

    // Get audio data from buffer or queue
    std::vector<float> source_audio(num_samples, 0.0f);

    if (config.is_looping) {
        // Looping source
        for (size_t i = 0; i < num_samples; ++i) {
            if (source->buffer_position >= source->audio_buffer.size()) {
                source->buffer_position = 0;
            }
            if (source->buffer_position < source->audio_buffer.size()) {
                source_audio[i] = source->audio_buffer[source->buffer_position++];
            }
        }
    } else {
        // Non-looping source - consume from queue if available
        {
            std::lock_guard<std::mutex> queue_lock(source->queue_mutex);
            for (size_t i = 0; i < num_samples && !source->audio_queue.empty(); ++i) {
                source_audio[i] = source->audio_queue.front();
                source->audio_queue.pop();
            }
        }
    }

    // Apply gain
    for (size_t i = 0; i < num_samples; ++i) {
        source_audio[i] *= source->current_gain;
    }

    // Apply spatial processing based on rendering mode
    if (config.is_spatialized) {
        switch (rendering_mode_) {
            case VRRenderingMode::BINAURAL:
                applyBinauralProcessing(source, source_audio.data(), output_buffer, num_samples);
                break;
            case VRRenderingMode::STEREO_PANNING:
                applyStereoPanning(source, source_audio.data(), output_buffer, num_samples);
                break;
            default:
                // Default to binaural
                applyBinauralProcessing(source, source_audio.data(), output_buffer, num_samples);
                break;
        }
    } else {
        // No spatial processing - just mix directly
        for (size_t i = 0; i < num_samples; ++i) {
            for (int ch = 0; ch < channels_; ++ch) {
                output_buffer[i * channels_ + ch] += source_audio[i];
            }
        }
    }
}

void VRAudioProcessor::applyBinauralProcessing(VRAudioSourceState* source, const float* input,
                                              float* output, size_t num_samples) {
    // Convert source position to listener-relative coordinates
    float relative_pos[3];
    vr_audio_utils::worldToListenerSpace(source->config.position, current_head_pose_.position,
                                        current_head_pose_.orientation, relative_pos);

    // Calculate azimuth and elevation for HRTF selection
    float distance = std::sqrt(relative_pos[0] * relative_pos[0] +
                              relative_pos[1] * relative_pos[1] +
                              relative_pos[2] * relative_pos[2]);

    if (distance < MIN_DISTANCE) {
        distance = MIN_DISTANCE;
    }

    float azimuth = std::atan2(relative_pos[0], relative_pos[2]) * 180.0f / M_PI;
    float elevation = std::asin(relative_pos[1] / distance) * 180.0f / M_PI;

    // Get HRTF impulse responses
    auto hrtf_irs = getHRTFImpulseResponse(relative_pos, current_head_pose_.position,
                                          current_head_pose_.orientation);

    if (hrtf_irs.first.empty() || hrtf_irs.second.empty()) {
        // Fall back to simple panning if HRTF unavailable
        applyStereoPanning(source, input, output, num_samples);
        return;
    }

    // Apply HRTF convolution (simplified - real implementation would use FFT-based convolution)
    for (size_t i = 0; i < num_samples; ++i) {
        float left_sample = 0.0f;
        float right_sample = 0.0f;

        // Simple time-domain convolution (not efficient for real production use)
        for (size_t j = 0; j < hrtf_irs.first.size() && j <= i; ++j) {
            left_sample += input[i - j] * hrtf_irs.first[j];
            right_sample += input[i - j] * hrtf_irs.second[j];
        }

        output[i * 2] += left_sample;     // Left channel
        output[i * 2 + 1] += right_sample; // Right channel
    }
}

void VRAudioProcessor::applyStereoPanning(VRAudioSourceState* source, const float* input,
                                         float* output, size_t num_samples) {
    // Calculate stereo panning based on source position
    float relative_pos[3];
    vr_audio_utils::worldToListenerSpace(source->config.position, current_head_pose_.position,
                                        current_head_pose_.orientation, relative_pos);

    // Calculate azimuth for panning
    float azimuth = std::atan2(relative_pos[0], relative_pos[2]);

    // Simple pan law: sin/cos panning
    float pan_left = std::cos((azimuth + M_PI/2) * 0.5f);
    float pan_right = std::sin((azimuth + M_PI/2) * 0.5f);

    // Normalize to maintain constant power
    float normalization = std::sqrt(pan_left * pan_left + pan_right * pan_right);
    if (normalization > 0.0f) {
        pan_left /= normalization;
        pan_right /= normalization;
    }

    // Apply panning
    for (size_t i = 0; i < num_samples; ++i) {
        output[i * 2] += input[i] * pan_left;      // Left channel
        output[i * 2 + 1] += input[i] * pan_right;  // Right channel
    }
}

void VRAudioProcessor::applyReverb(float* buffer, size_t num_samples) {
    if (!room_acoustics_.enable_late_reverb && !room_acoustics_.enable_early_reflections) {
        return;
    }

    // Simple reverb implementation - in production, would use more sophisticated algorithms
    static std::vector<float> reverb_buffer(16384, 0.0f);
    static size_t reverb_buffer_pos = 0;

    // Early reflections
    if (room_acoustics_.enable_early_reflections) {
        // Apply simple early reflection delays
        for (size_t i = 0; i < num_samples; ++i) {
            float sum = buffer[i * 2] + buffer[i * 2 + 1];

            // Add delayed versions at different gains
            size_t delay_1 = static_cast<size_t>(0.03f * sample_rate_); // 30ms delay
            size_t delay_2 = static_cast<size_t>(0.05f * sample_rate_); // 50ms delay

            size_t pos_1 = (reverb_buffer_pos + delay_1) % reverb_buffer.size();
            size_t pos_2 = (reverb_buffer_pos + delay_2) % reverb_buffer.size();

            float reflection = reverb_buffer[pos_1] * 0.3f + reverb_buffer[pos_2] * 0.2f;

            buffer[i * 2] += reflection * room_acoustics_.reflection_coefficient;
            buffer[i * 2 + 1] += reflection * room_acoustics_.reflection_coefficient;

            // Update reverb buffer
            reverb_buffer[reverb_buffer_pos] = sum * 0.5f;
            reverb_buffer_pos = (reverb_buffer_pos + 1) % reverb_buffer.size();
        }
    }
}

std::pair<std::vector<float>, std::vector<float>> VRAudioProcessor::getHRTFImpulseResponse(
    const float source_pos[3], const float listener_pos[3], const float listener_orientation[4]) {

    // Generate cache key from positions
    uint64_t cache_key = 0;
    cache_key ^= static_cast<uint64_t>(std::hash<int>{}(static_cast<int>(source_pos[0] * 100)) << 0);
    cache_key ^= static_cast<uint64_t>(std::hash<int>{}(static_cast<int>(source_pos[1] * 100)) << 20);
    cache_key ^= static_cast<uint64_t>(std::hash<int>{}(static_cast<int>(source_pos[2] * 100)) << 40);

    // Check cache first
    {
        std::lock_guard<std::mutex> lock(hrtf_mutex_);
        auto it = hrtf_cache_.find(cache_key);
        if (it != hrtf_cache_.end()) {
            metrics_.hrtf_cache_hits++;
            return it->second;
        }
        metrics_.hrtf_cache_misses++;
    }

    // Generate HRTF impulse responses
    std::vector<float> left_ir(quality_settings_.hrtf_impulse_response_length);
    std::vector<float> right_ir(quality_settings_.hrtf_impulse_response_length);

    // Calculate relative position and angles
    float relative_pos[3];
    vr_audio_utils::worldToListenerSpace(source_pos, listener_pos, listener_orientation, relative_pos);

    float distance = std::sqrt(relative_pos[0] * relative_pos[0] +
                              relative_pos[1] * relative_pos[1] +
                              relative_pos[2] * relative_pos[2]);

    float azimuth = std::atan2(relative_pos[0], relative_pos[2]);
    float elevation = std::asin(std::max(-1.0f, std::min(1.0f, relative_pos[1] / std::max(distance, MIN_DISTANCE))));

    // Generate simplified HRTF using ITD and ILD cues
    float itd = ear_distance_ * std::sin(azimuth) / DEFAULT_SOUND_SPEED;
    int itd_samples = static_cast<int>(itd * sample_rate_);

    // Generate impulse responses with ITD and spectral shaping
    for (int i = 0; i < quality_settings_.hrtf_impulse_response_length; ++i) {
        float t = static_cast<float>(i) / sample_rate_;

        // Basic pinna transfer function approximation
        float left_filter = 1.0f + 0.3f * std::cos(2.0f * M_PI * 3000.0f * t); // Simple high-frequency boost
        float right_filter = 1.0f + 0.3f * std::cos(2.0f * M_PI * 2500.0f * t);

        // Apply ITD delay
        if (i >= itd_samples && i < itd_samples + 100) {
            left_ir[i] = 0.0f;
            right_ir[i] = 0.0f;
        } else if (i < 100) {
            left_ir[i] = left_filter * std::exp(-t * 10.0f);
            right_ir[i] = right_filter * std::exp(-t * 10.0f);
        } else {
            left_ir[i] = 0.0f;
            right_ir[i] = 0.0f;
        }
    }

    // Add to cache
    {
        std::lock_guard<std::mutex> lock(hrtf_mutex_);
        if (hrtf_cache_.size() < HRTF_CACHE_MAX_SIZE) {
            hrtf_cache_[cache_key] = {left_ir, right_ir};
        }
    }

    return {left_ir, right_ir};
}

void VRAudioProcessor::updateMetrics() {
    auto now = std::chrono::high_resolution_clock::now();

    std::lock_guard<std::mutex> lock(metrics_mutex_);

    // Count active sources
    metrics_.active_sources = 0;
    metrics_.rendered_sources = 0;

    {
        std::lock_guard<std::mutex> sources_lock(sources_mutex_);
        for (const auto& [id, state] : audio_sources_) {
            if (state->is_active) {
                metrics_.active_sources++;
                metrics_.rendered_sources++;
            }
        }
    }

    // Update HRTF cache stats
    auto [hit_ratio, cache_size] = getHRTFCacheStats();
    metrics_.hrtf_cache_hit_ratio = hit_ratio;

    // Calculate average latency (simplified)
    metrics_.average_latency_ms = metrics_.processing_time_ms;
    metrics_.peak_latency_ms = std::max(metrics_.peak_latency_ms, metrics_.processing_time_ms);

    // Call metrics callback if set
    if (metrics_callback_) {
        metrics_callback_(metrics_);
    }
}

void VRAudioProcessor::adaptQuality() {
    auto now = std::chrono::high_resolution_clock::now();

    // Check if enough time has passed since last adaptation
    auto time_since_adaptation = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_quality_adaptation_).count();

    if (time_since_adaptation < 1000) { // Adapt at most once per second
        return;
    }

    // Track recent processing times
    recent_processing_times_.push_back(metrics_.processing_time_ms);
    if (recent_processing_times_.size() > PROCESSING_TIME_HISTORY) {
        recent_processing_times_.erase(recent_processing_times_.begin());
    }

    if (recent_processing_times_.size() < PROCESSING_TIME_HISTORY / 2) {
        return; // Not enough data yet
    }

    // Calculate average processing time
    double avg_processing_time = 0.0;
    for (double time : recent_processing_times_) {
        avg_processing_time += time;
    }
    avg_processing_time /= recent_processing_times_.size();

    // Determine if quality adjustment is needed
    int new_quality_level = current_quality_level_;

    if (avg_processing_time > target_latency_ms_ * QUALITY_ADAPTATION_THRESHOLD) {
        // Processing too slow - reduce quality
        if (current_quality_level_ > static_cast<int>(min_quality_level_)) {
            new_quality_level = current_quality_level_ - 1;
        }
    } else if (avg_processing_time < target_latency_ms_ * 0.5) {
        // Processing fast enough - can increase quality
        if (current_quality_level_ < static_cast<int>(VRRenderingQuality::ULTRA_HIGH)) {
            new_quality_level = current_quality_level_ + 1;
        }
    }

    // Apply quality change if needed
    if (new_quality_level != current_quality_level_) {
        current_quality_level_ = new_quality_level;
        metrics_.quality_level = new_quality_level;
        metrics_.quality_adaptation_active = true;

        // Update quality settings based on new level
        switch (static_cast<VRRenderingQuality::Level>(new_quality_level)) {
            case VRRenderingQuality::ULTRA_HIGH:
                quality_settings_.hrtf_quality = VRRenderingQuality::ULTRA_HIGH;
                quality_settings_.reverb_quality = VRRenderingQuality::HIGH;
                quality_settings_.hrtf_impulse_response_length = 512;
                break;
            case VRRenderingQuality::HIGH:
                quality_settings_.hrtf_quality = VRRenderingQuality::HIGH;
                quality_settings_.reverb_quality = VRRenderingQuality::MEDIUM;
                quality_settings_.hrtf_impulse_response_length = 256;
                break;
            case VRRenderingQuality::MEDIUM:
                quality_settings_.hrtf_quality = VRRenderingQuality::MEDIUM;
                quality_settings_.reverb_quality = VRRenderingQuality::LOW;
                quality_settings_.hrtf_impulse_response_length = 128;
                break;
            case VRRenderingQuality::LOW:
                quality_settings_.hrtf_quality = VRRenderingQuality::LOW;
                quality_settings_.reverb_quality = VRRenderingQuality::MINIMAL;
                quality_settings_.hrtf_impulse_response_length = 64;
                break;
            case VRRenderingQuality::MINIMAL:
                quality_settings_.hrtf_quality = VRRenderingQuality::MINIMAL;
                quality_settings_.reverb_quality = VRRenderingQuality::MINIMAL;
                quality_settings_.hrtf_impulse_response_length = 32;
                break;
        }

        last_quality_adaptation_ = now;
    } else {
        metrics_.quality_adaptation_active = false;
    }
}

bool VRAudioProcessor::validateSource(const VRAudioSource& source) const {
    return source.min_distance > 0.0f &&
           source.max_distance > source.min_distance &&
           source.reference_distance > 0.0f &&
           source.rolloff_factor >= 0.0f &&
           source.gain >= 0.0f &&
           source.directivity >= 0.0f && source.directivity <= 1.0f;
}

void VRAudioProcessor::calculateDistanceAttenuation(const VRAudioSource& source, float distance, float& gain) {
    switch (source.distance_model) {
        case AudioDistanceModel::NONE:
            gain = 1.0f;
            break;

        case AudioDistanceModel::INVERSE:
            gain = source.reference_distance / (source.reference_distance +
                   source.rolloff_factor * (distance - source.reference_distance));
            break;

        case AudioDistanceModel::INVERSE_CLAMPED:
            if (distance < source.reference_distance) {
                gain = 1.0f;
            } else if (distance > source.max_distance) {
                gain = 0.0f;
            } else {
                gain = source.reference_distance / (source.reference_distance +
                       source.rolloff_factor * (distance - source.reference_distance));
            }
            break;

        case AudioDistanceModel::EXPONENTIAL:
            gain = std::pow(distance / source.reference_distance, -source.rolloff_factor);
            break;

        case AudioDistanceModel::LINEAR:
            gain = 1.0f - source.rolloff_factor * (distance - source.reference_distance) /
                   (source.max_distance - source.reference_distance);
            break;

        default:
            gain = 1.0f; // Default to no attenuation
            break;
    }

    gain = std::max(0.0f, std::min(1.0f, gain));
}

void VRAudioProcessor::calculateDirectivityGain(const VRAudioSource& source, const float listener_pos[3], float& gain) {
    if (source.directivity <= 0.0f) {
        gain = 1.0f; // Omnidirectional
        return;
    }

    // Calculate direction from source to listener
    float direction[3] = {
        listener_pos[0] - source.position[0],
        listener_pos[1] - source.position[1],
        listener_pos[2] - source.position[2]
    };

    float distance = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
    if (distance < MIN_DISTANCE) {
        gain = 1.0f;
        return;
    }

    // Normalize direction
    direction[0] /= distance;
    direction[1] /= distance;
    direction[2] /= distance;

    // Get source forward direction from orientation quaternion
    float source_forward[3] = {0.0f, 0.0f, 1.0f};
    vr_audio_utils::rotateVector(source.orientation, source_forward);

    // Calculate dot product for angle
    float dot_product = direction[0] * source_forward[0] +
                       direction[1] * source_forward[1] +
                       direction[2] * source_forward[2];

    // Clamp to valid range
    dot_product = std::max(-1.0f, std::min(1.0f, dot_product));

    float angle = std::acos(dot_product);

    // Convert cone angles to radians
    float inner_cone_rad = source.inner_cone_angle * M_PI / 180.0f;
    float outer_cone_rad = source.outer_cone_angle * M_PI / 180.0f;

    if (angle <= inner_cone_rad) {
        gain = 1.0f; // Inside inner cone
    } else if (angle >= outer_cone_rad) {
        gain = source.outer_cone_gain; // Outside outer cone
    } else {
        // Between cones - interpolate gain
        float interpolation = (angle - inner_cone_rad) / (outer_cone_rad - inner_cone_rad);
        gain = 1.0f + interpolation * (source.outer_cone_gain - 1.0f);
    }

    // Apply directivity factor
    gain = gain * (1.0f - source.directivity) + source.directivity * gain;
}

// VRAudioFactory implementation
std::unique_ptr<VRAudioProcessor> VRAudioFactory::createVRProcessor(
    const std::string& use_case, int sample_rate, int buffer_size) {

    VRRenderingQuality quality;

    if (use_case == "gaming" || use_case == "game") {
        return createGamingVRProcessor(sample_rate, buffer_size);
    } else if (use_case == "social" || use_case == "vrchat" || use_case == "recroom") {
        return createSocialVRProcessor(sample_rate, buffer_size);
    } else if (use_case == "professional" || use_case == "training" || use_case == "simulation") {
        return createProfessionalVRProcessor(sample_rate, buffer_size);
    } else {
        // Default balanced settings
        quality = VRRenderingQuality{};
        quality.hrtf_quality = VRRenderingQuality::MEDIUM;
        quality.reverb_quality = VRRenderingQuality::MEDIUM;
        quality.enable_gpu_acceleration = true;

        auto processor = std::make_unique<VRAudioProcessor>();
        if (processor->initialize(sample_rate, buffer_size, 2, quality)) {
            return processor;
        }
        return nullptr;
    }
}

std::unique_ptr<VRAudioProcessor> VRAudioFactory::createGamingVRProcessor(int sample_rate, int buffer_size) {
    VRRenderingQuality quality;
    quality.hrtf_quality = VRRenderingQuality::MEDIUM;
    quality.reverb_quality = VRRenderingQuality::LOW;
    quality.occlusion_quality = VRRenderingQuality::LOW;
    quality.reflection_quality = VRRenderingQuality::MINIMAL;
    quality.enable_gpu_acceleration = true;
    quality.enable_multi_threading = true;
    quality.max_concurrent_sources = 64;
    quality.processing_latency_target = 8.0f;
    quality.hrtf_impulse_response_length = 128;

    auto processor = std::make_unique<VRAudioProcessor>();
    if (processor->initialize(sample_rate, buffer_size, 2, quality)) {
        return processor;
    }
    return nullptr;
}

std::unique_ptr<VRAudioProcessor> VRAudioFactory::createSocialVRProcessor(int sample_rate, int buffer_size) {
    VRRenderingQuality quality;
    quality.hrtf_quality = VRRenderingQuality::HIGH;
    quality.reverb_quality = VRRenderingQuality::MEDIUM;
    quality.occlusion_quality = VRRenderingQuality::MEDIUM;
    quality.reflection_quality = VRRenderingQuality::LOW;
    quality.enable_gpu_acceleration = true;
    quality.enable_multi_threading = true;
    quality.max_concurrent_sources = 32;
    quality.processing_latency_target = 15.0f;
    quality.hrtf_impulse_response_length = 256;

    auto processor = std::make_unique<VRAudioProcessor>();
    if (processor->initialize(sample_rate, buffer_size, 2, quality)) {
        return processor;
    }
    return nullptr;
}

std::unique_ptr<VRAudioProcessor> VRAudioFactory::createProfessionalVRProcessor(int sample_rate, int buffer_size) {
    VRRenderingQuality quality;
    quality.hrtf_quality = VRRenderingQuality::ULTRA_HIGH;
    quality.reverb_quality = VRRenderingQuality::HIGH;
    quality.occlusion_quality = VRRenderingQuality::HIGH;
    quality.reflection_quality = VRRenderingQuality::MEDIUM;
    quality.enable_gpu_acceleration = true;
    quality.enable_multi_threading = true;
    quality.max_concurrent_sources = 128;
    quality.processing_latency_target = 25.0f;
    quality.hrtf_impulse_response_length = 512;

    auto processor = std::make_unique<VRAudioProcessor>();
    if (processor->initialize(sample_rate, buffer_size, 2, quality)) {
        return processor;
    }
    return nullptr;
}

VRRenderingQuality VRAudioFactory::getRecommendedQuality(const std::string& platform, float target_fps) {
    VRRenderingQuality quality;

    // Platform-specific optimizations
    if (platform == "quest" || platform == "oculus" || platform == "meta") {
        // Meta Quest optimizations
        if (target_fps >= 90.0f) {
            quality.hrtf_quality = VRRenderingQuality::MEDIUM;
            quality.reverb_quality = VRRenderingQuality::LOW;
            quality.processing_latency_target = 8.0f;
        } else {
            quality.hrtf_quality = VRRenderingQuality::LOW;
            quality.reverb_quality = VRRenderingQuality::MINIMAL;
            quality.processing_latency_target = 10.0f;
        }
        quality.enable_gpu_acceleration = true;
        quality.max_concurrent_sources = 32;

    } else if (platform == "vive" || platform == "index" || platform == "pcvr") {
        // PC VR optimizations - more headroom
        quality.hrtf_quality = VRRenderingQuality::HIGH;
        quality.reverb_quality = VRRenderingQuality::MEDIUM;
        quality.enable_gpu_acceleration = true;
        quality.max_concurrent_sources = 64;
        quality.processing_latency_target = 12.0f;

    } else if (platform == "pico" || platform == "pico4") {
        // Pico optimizations
        quality.hrtf_quality = VRRenderingQuality::MEDIUM;
        quality.reverb_quality = VRRenderingQuality::LOW;
        quality.enable_gpu_acceleration = true;
        quality.max_concurrent_sources = 24;
        quality.processing_latency_target = 10.0f;

    } else {
        // Default balanced settings
        quality.hrtf_quality = VRRenderingQuality::MEDIUM;
        quality.reverb_quality = VRRenderingQuality::MEDIUM;
        quality.enable_gpu_acceleration = true;
        quality.max_concurrent_sources = 32;
        quality.processing_latency_target = 15.0f;
    }

    return quality;
}

// Utility functions implementation
namespace vr_audio_utils {

void worldToListenerSpace(const float world_pos[3], const float listener_pos[3],
                         const float listener_orientation[4], float result[3]) {
    // Translate to listener-relative coordinates
    result[0] = world_pos[0] - listener_pos[0];
    result[1] = world_pos[1] - listener_pos[1];
    result[2] = world_pos[2] - listener_pos[2];

    // Rotate by inverse of listener orientation
    float inverse_orientation[4] = {
        -listener_orientation[0],
        -listener_orientation[1],
        -listener_orientation[2],
        listener_orientation[3]
    };

    rotateVector(inverse_orientation, result);
}

void listenerToWorldSpace(const float listener_pos[3], const float listener_orientation[4],
                        float result[3]) {
    // This would implement the inverse transformation
    // For now, just copy the listener position
    result[0] = listener_pos[0];
    result[1] = listener_pos[1];
    result[2] = listener_pos[2];
}

float calculateDistance(const float pos1[3], const float pos2[3]) {
    float dx = pos1[0] - pos2[0];
    float dy = pos1[1] - pos2[1];
    float dz = pos1[2] - pos2[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

float calculateRelativeVelocity(const float vel1[3], const float vel2[3],
                              const float direction[3]) {
    float rel_vel[3] = {
        vel1[0] - vel2[0],
        vel1[1] - vel2[1],
        vel1[2] - vel2[2]
    };

    return rel_vel[0] * direction[0] + rel_vel[1] * direction[1] + rel_vel[2] * direction[2];
}

float calculateDopplerShift(float source_velocity, float listener_velocity,
                           float sound_speed, float relative_distance_change) {
    float effective_velocity = source_velocity - listener_velocity;

    if (std::abs(effective_velocity) < 0.001f) {
        return 1.0f; // No significant relative motion
    }

    // Simple Doppler formula: f' = f * (c + v_l) / (c - v_s)
    float doppler_factor = (sound_speed + listener_velocity) / (sound_speed - source_velocity);

    // Clamp to reasonable range
    return std::max(0.5f, std::min(2.0f, doppler_factor));
}

float calculateAirAbsorption(float distance, float frequency, float temperature, float humidity) {
    // Simplified air absorption model
    // Real implementation would use more complex formulas from ISO 9613-1

    // Base absorption coefficient (simplified)
    float absorption_coeff = 0.01f * frequency / 1000.0f; // Increases with frequency

    // Temperature and humidity corrections
    absorption_coeff *= (1.0f + 0.01f * (temperature - 20.0f)); // Temperature effect
    absorption_coeff *= (1.0f - 0.002f * (humidity - 50.0f));   // Humidity effect

    // Calculate attenuation over distance
    float attenuation = std::exp(-absorption_coeff * distance);

    return attenuation;
}

bool isPointInRoom(const float point[3], const float room_dims[3]) {
    return point[0] >= 0.0f && point[0] <= room_dims[0] &&
           point[1] >= 0.0f && point[1] <= room_dims[1] &&
           point[2] >= 0.0f && point[2] <= room_dims[2];
}

float calculateRoomVolume(const float room_dims[3]) {
    return room_dims[0] * room_dims[1] * room_dims[2];
}

float calculateSurfaceArea(const float room_dims[3]) {
    return 2.0f * (room_dims[0] * room_dims[1] +
                   room_dims[0] * room_dims[2] +
                   room_dims[1] * room_dims[2]);
}

VRRenderingQuality calculateOptimalQuality(double cpu_usage, double gpu_usage,
                                         float target_latency, int source_count) {
    VRRenderingQuality quality;

    // Adaptive quality calculation based on system resources

    if (cpu_usage > 80.0 || gpu_usage > 80.0) {
        // System under heavy load - reduce quality
        quality.hrtf_quality = VRRenderingQuality::LOW;
        quality.reverb_quality = VRRenderingQuality::MINIMAL;
        quality.occlusion_quality = VRRenderingQuality::MINIMAL;
        quality.reflection_quality = VRRenderingQuality::MINIMAL;
        quality.max_concurrent_sources = std::min(16, source_count);
        quality.hrtf_impulse_response_length = 64;

    } else if (cpu_usage > 60.0 || gpu_usage > 60.0) {
        // Moderate load - medium quality
        quality.hrtf_quality = VRRenderingQuality::MEDIUM;
        quality.reverb_quality = VRRenderingQuality::LOW;
        quality.occlusion_quality = VRRenderingQuality::LOW;
        quality.reflection_quality = VRRenderingQuality::MINIMAL;
        quality.max_concurrent_sources = std::min(32, source_count);
        quality.hrtf_impulse_response_length = 128;

    } else if (target_latency > 20.0f) {
        // Higher latency tolerance - can use higher quality
        quality.hrtf_quality = VRRenderingQuality::HIGH;
        quality.reverb_quality = VRRenderingQuality::MEDIUM;
        quality.occlusion_quality = VRRenderingQuality::MEDIUM;
        quality.reflection_quality = VRRenderingQuality::LOW;
        quality.max_concurrent_sources = std::min(64, source_count);
        quality.hrtf_impulse_response_length = 256;

    } else {
        // Low load and latency - high quality
        quality.hrtf_quality = VRRenderingQuality::HIGH;
        quality.reverb_quality = VRRenderingQuality::MEDIUM;
        quality.occlusion_quality = VRRenderingQuality::MEDIUM;
        quality.reflection_quality = VRRenderingQuality::LOW;
        quality.max_concurrent_sources = std::min(48, source_count);
        quality.hrtf_impulse_response_length = 256;
    }

    quality.enable_gpu_acceleration = true;
    quality.enable_multi_threading = true;

    return quality;
}

bool isQualityLevelAcceptable(VRRenderingQuality::Level level,
                             const VRRenderingQuality& settings) {
    switch (level) {
        case VRRenderingQuality::ULTRA_HIGH:
            return settings.hrtf_quality == VRRenderingQuality::ULTRA_HIGH &&
                   settings.reverb_quality >= VRRenderingQuality::HIGH;

        case VRRenderingQuality::HIGH:
            return settings.hrtf_quality >= VRRenderingQuality::HIGH &&
                   settings.reverb_quality >= VRRenderingQuality::MEDIUM;

        case VRRenderingQuality::MEDIUM:
            return settings.hrtf_quality >= VRRenderingQuality::MEDIUM &&
                   settings.reverb_quality >= VRRenderingQuality::LOW;

        case VRRenderingQuality::LOW:
            return settings.hrtf_quality >= VRRenderingQuality::LOW &&
                   settings.reverb_quality >= VRRenderingQuality::MINIMAL;

        case VRRenderingQuality::MINIMAL:
            return true; // Always acceptable
    }

    return false;
}

} // namespace vr_audio_utils

} // namespace dsp
} // namespace core
} // namespace vortex