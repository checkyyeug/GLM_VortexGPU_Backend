#include "core/audio/audio_routing.hpp"
#include "core/dsp/audio_math.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <chrono>
#include <random>
#include <stdexcept>

namespace vortex {
namespace core {
namespace audio {

// Constants
constexpr size_t CACHE_LINE_SIZE = 64;
constexpr int ALIGNMENT = 64;
constexpr int MAX_DELAY_SECONDS = 10;
constexpr float GAIN_SMOOTHING_TIME_S = 0.001f; // 1ms
constexpr float MIN_GAIN = 0.0f;
constexpr float MAX_GAIN = 100.0f;
constexpr float MIN_PAN = -1.0f;
constexpr float MAX_PAN = 1.0f;
constexpr int METRICS_UPDATE_INTERVAL_FRAMES = 1000;
constexpr double MAX_ACCEPTABLE_XRUN_RATE = 0.01; // 1% XRUN rate acceptable

// LockFreeRingBuffer implementation
LockFreeRingBuffer::LockFreeRingBuffer(size_t size) : size_(size) {
    if (size == 0 || (size & (size - 1)) != 0) {
        throw std::invalid_argument("Ring buffer size must be power of 2");
    }

    mask_ = size - 1;
    buffer_ = std::make_unique<float[]>(size);
    std::memset(buffer_.get(), 0, size * sizeof(float));
}

LockFreeRingBuffer::~LockFreeRingBuffer() = default;

bool LockFreeRingBuffer::write(const float* data, size_t samples) {
    if (!data || samples == 0) {
        return false;
    }

    size_t write_pos = write_pos_.load(std::memory_order_relaxed);
    size_t read_pos = read_pos_.load(std::memory_order_acquire);

    size_t available_space = size_ - (write_pos - read_pos);
    if (samples > available_space) {
        return false; // Not enough space
    }

    for (size_t i = 0; i < samples; ++i) {
        buffer_[(write_pos + i) & mask_] = data[i];
    }

    write_pos_.store(write_pos + samples, std::memory_order_release);
    return true;
}

bool LockFreeRingBuffer::read(float* data, size_t samples) {
    if (!data || samples == 0) {
        return false;
    }

    size_t read_pos = read_pos_.load(std::memory_order_relaxed);
    size_t write_pos = write_pos_.load(std::memory_order_acquire);

    size_t available_data = write_pos - read_pos;
    if (samples > available_data) {
        return false; // Not enough data
    }

    for (size_t i = 0; i < samples; ++i) {
        data[i] = buffer_[(read_pos + i) & mask_];
    }

    read_pos_.store(read_pos + samples, std::memory_order_release);
    return true;
}

size_t LockFreeRingBuffer::available() const {
    size_t write_pos = write_pos_.load(std::memory_order_relaxed);
    size_t read_pos = read_pos_.load(std::memory_order_relaxed);
    return write_pos - read_pos;
}

size_t LockFreeRingBuffer::space() const {
    size_t write_pos = write_pos_.load(std::memory_order_relaxed);
    size_t read_pos = read_pos_.load(std::memory_order_relaxed);
    return size_ - (write_pos - read_pos);
}

void LockFreeRingBuffer::clear() {
    read_pos_.store(write_pos_.load(std::memory_order_relaxed), std::memory_order_release);
}

// AudioProcessorNode implementation
AudioProcessorNode::AudioProcessorNode(uint32_t id, RoutingNode type, int input_channels, int output_channels)
    : id_(id), type_(type), input_channels_(input_channels), output_channels_(output_channels),
      is_active_(true), bypassed_(false) {
}

// GainNode implementation
GainNode::GainNode(uint32_t id, int channels)
    : AudioProcessorNode(id, RoutingNode::GAIN, channels, channels) {
    parameters_.resize(1, 1.0f); // gain parameter
    parameter_names_ = {"Gain"};
}

bool GainNode::initialize(const RoutingConfig& config) {
    config_ = config;
    gain_ = 1.0f;
    target_gain_ = 1.0f;
    ramping_ = false;
    return true;
}

void GainNode::process(const float** inputs, float** outputs, size_t samples) {
    if (bypassed_ || !inputs || !outputs) {
        // Pass through if bypassed
        if (inputs && outputs && inputs[0] && outputs[0]) {
            std::copy(inputs[0], inputs[0] + samples, outputs[0]);
        }
        return;
    }

    float current_gain = gain_;
    float gain_diff = target_gain_ - current_gain;

    if (std::abs(gain_diff) < 0.0001f) {
        // No gain change needed
        for (int ch = 0; ch < input_channels_; ++ch) {
            if (inputs[ch] && outputs[ch]) {
                for (size_t i = 0; i < samples; ++i) {
                    outputs[ch][i] = inputs[ch][i] * target_gain_;
                }
            }
        }
        gain_ = target_gain_;
        ramping_ = false;
    } else {
        // Ramp gain
        float gain_step = gain_diff / static_cast<float>(samples);

        for (int ch = 0; ch < input_channels_; ++ch) {
            if (inputs[ch] && outputs[ch]) {
                current_gain = gain_;
                for (size_t i = 0; i < samples; ++i) {
                    current_gain += gain_step;
                    outputs[ch][i] = inputs[ch][i] * current_gain;
                }
            }
        }
        gain_ = current_gain;

        if (std::abs(gain_ - target_gain_) < 0.0001f) {
            gain_ = target_gain_;
            ramping_ = false;
        }
    }
}

void GainNode::reset() {
    gain_ = target_gain_ = 1.0f;
    ramping_ = false;
}

void GainNode::setParameter(int index, float value) {
    if (index == 0) {
        target_gain_ = std::max(MIN_GAIN, std::min(MAX_GAIN, value));
        ramping_ = true;
    }
}

float GainNode::getParameter(int index) const {
    if (index == 0) {
        return target_gain_;
    }
    return 0.0f;
}

// MixerNode implementation
MixerNode::MixerNode(uint32_t id, int input_channels, int output_channels)
    : AudioProcessorNode(id, RoutingNode::MIXER, input_channels, output_channels) {
    input_gains_.resize(input_channels, 1.0f);
    output_gains_.resize(output_channels, 1.0f);
    output_pans_.resize(output_channels, 0.0f);
}

bool MixerNode::initialize(const RoutingConfig& config) {
    config_ = config;
    std::fill(input_gains_.begin(), input_gains_.end(), 1.0f);
    std::fill(output_gains_.begin(), output_gains_.end(), 1.0f);
    std::fill(output_pans_.begin(), output_pans_.end(), 0.0f);
    return true;
}

void MixerNode::process(const float** inputs, float** outputs, size_t samples) {
    if (bypassed_) {
        // Pass through
        for (int ch = 0; ch < std::min(input_channels_, output_channels_); ++ch) {
            if (inputs[ch] && outputs[ch]) {
                std::copy(inputs[ch], inputs[ch] + samples, outputs[ch]);
            }
        }
        return;
    }

    // Clear outputs
    for (int out_ch = 0; out_ch < output_channels_; ++out_ch) {
        if (outputs[out_ch]) {
            std::memset(outputs[out_ch], 0, samples * sizeof(float));
        }
    }

    // Mix inputs to outputs
    for (int in_ch = 0; in_ch < input_channels_; ++in_ch) {
        if (!inputs[in_ch]) continue;

        float input_gain = input_gains_[in_ch];

        for (int out_ch = 0; out_ch < output_channels_; ++out_ch) {
            if (!outputs[out_ch]) continue;

            float output_gain = output_gains_[out_ch];
            float pan = output_pans_[out_ch];

            // Calculate stereo panning for stereo output
            float left_gain = 1.0f, right_gain = 1.0f;
            if (output_channels_ == 2) {
                left_gain = std::cos((pan + 1.0f) * M_PI / 4.0f);
                right_gain = std::sin((pan + 1.0f) * M_PI / 4.0f);
            }

            float total_gain = input_gain * output_gain;
            if (output_channels_ == 2) {
                total_gain *= (out_ch == 0) ? left_gain : right_gain;
            }

            for (size_t i = 0; i < samples; ++i) {
                outputs[out_ch][i] += inputs[in_ch][i] * total_gain;
            }
        }
    }
}

void MixerNode::reset() {
    // Nothing to reset for static mixer
}

void MixerNode::setParameter(int index, float value) {
    int param_index = index / 3; // 3 parameters per channel: gain, gain, pan
    int param_type = index % 3;

    if (param_type == 0 && param_index < input_channels_) {
        input_gains_[param_index] = value;
    } else if (param_type == 1 && param_index < output_channels_) {
        output_gains_[param_index] = value;
    } else if (param_type == 2 && param_index < output_channels_) {
        output_pans_[param_index] = std::max(MIN_PAN, std::min(MAX_PAN, value));
    }
}

float MixerNode::getParameter(int index) const {
    int param_index = index / 3;
    int param_type = index % 3;

    if (param_type == 0 && param_index < input_channels_) {
        return input_gains_[param_index];
    } else if (param_type == 1 && param_index < output_channels_) {
        return output_gains_[param_index];
    } else if (param_type == 2 && param_index < output_channels_) {
        return output_pans_[param_index];
    }

    return 0.0f;
}

void MixerNode::setInputGain(int input, float gain) {
    if (input >= 0 && input < input_channels_) {
        input_gains_[input] = gain;
    }
}

void MixerNode::setOutputGain(int output, float gain) {
    if (output >= 0 && output < output_channels_) {
        output_gains_[output] = gain;
    }
}

void MixerNode::setPan(int output, float pan) {
    if (output >= 0 && output < output_channels_) {
        output_pans_[output] = std::max(MIN_PAN, std::min(MAX_PAN, pan));
    }
}

// DelayNode implementation
DelayNode::DelayNode(uint32_t id, int channels, int max_delay_samples)
    : AudioProcessorNode(id, RoutingNode::DELAY, channels, channels),
      max_delay_samples_(max_delay_samples) {
    parameters_.resize(1, 0.0f); // delay parameter
    parameter_names_ = {"Delay"};
}

bool DelayNode::initialize(const RoutingConfig& config) {
    config_ = config;

    // Create delay lines for each channel
    delay_lines_.clear();
    for (int ch = 0; ch < input_channels_; ++ch) {
        auto delay_line = std::make_unique<LockFreeRingBuffer>(max_delay_samples_);
        delay_line->clear();
        delay_lines_.push_back(std::move(delay_line));
    }

    delay_seconds_ = 0.0f;
    delay_samples_ = 0;

    return true;
}

void DelayNode::process(const float** inputs, float** outputs, size_t samples) {
    if (bypassed_) {
        // Pass through
        for (int ch = 0; ch < input_channels_; ++ch) {
            if (inputs[ch] && outputs[ch]) {
                std::copy(inputs[ch], inputs[ch] + samples, outputs[ch]);
            }
        }
        return;
    }

    for (int ch = 0; ch < input_channels_; ++ch) {
        if (!inputs[ch] || !outputs[ch] || ch >= static_cast<int>(delay_lines_.size())) {
            continue;
        }

        auto& delay_line = delay_lines_[ch];

        // Write input to delay line
        delay_line->write(inputs[ch], samples);

        // Read from delay line (with current delay setting)
        if (delay_samples_ > 0 && delay_samples_ <= max_delay_samples_) {
            size_t read_samples = std::min(static_cast<size_t>(delay_samples_), delay_line->available());
            size_t remaining_samples = samples - read_samples;

            // Read delayed samples
            if (read_samples > 0) {
                float* temp_buffer = new float[read_samples];
                if (delay_line->read(temp_buffer, read_samples)) {
                    std::copy(temp_buffer, temp_buffer + read_samples, outputs[ch]);
                } else {
                    std::memset(outputs[ch], 0, read_samples * sizeof(float));
                }
                delete[] temp_buffer;

                // Fill remaining samples with zeros
                if (remaining_samples > 0) {
                    std::memset(outputs[ch] + read_samples, 0, remaining_samples * sizeof(float));
                }
            } else {
                // No delayed samples available
                std::memset(outputs[ch], 0, samples * sizeof(float));
            }
        } else {
            // No delay - read directly
            delay_line->read(outputs[ch], samples);
        }
    }
}

void DelayNode::reset() {
    for (auto& delay_line : delay_lines_) {
        if (delay_line) {
            delay_line->clear();
        }
    }
    delay_samples_ = 0;
}

void DelayNode::setParameter(int index, float value) {
    if (index == 0) {
        delay_seconds_ = std::max(0.0f, std::min(static_cast<float>(MAX_DELAY_SECONDS), value));
        delay_samples_ = static_cast<int>(delay_seconds_ * config_.sample_rate);
        delay_samples_ = std::min(delay_samples_, max_delay_samples_);
    }
}

float DelayNode::getParameter(int index) const {
    if (index == 0) {
        return delay_seconds_;
    }
    return 0.0f;
}

void DelayNode::setDelay(float delay_seconds) {
    setParameter(0, delay_seconds);
}

float DelayNode::getDelay() const {
    return delay_seconds_;
}

// PanNode implementation
PanNode::PanNode(uint32_t id, int input_channels, int output_channels)
    : AudioProcessorNode(id, RoutingNode::PAN, input_channels, output_channels) {
    parameters_.resize(1, 0.0f); // pan parameter
    parameter_names_ = {"Pan"};
}

bool PanNode::initialize(const RoutingConfig& config) {
    config_ = config;
    pan_ = 0.0f;
    updateGains();
    return true;
}

void PanNode::process(const float** inputs, float** outputs, size_t samples) {
    if (bypassed_ || input_channels_ != 1 || output_channels_ != 2) {
        // Pass through for bypass or unsupported channel config
        for (int ch = 0; ch < std::min(input_channels_, output_channels_); ++ch) {
            if (inputs[ch] && outputs[ch]) {
                std::copy(inputs[ch], inputs[ch] + samples, outputs[ch]);
            }
        }
        return;
    }

    if (!inputs[0] || !outputs[0] || !outputs[1]) {
        return;
    }

    // Apply pan to stereo output
    for (size_t i = 0; i < samples; ++i) {
        outputs[0][i] = inputs[0][i] * left_gain_;
        outputs[1][i] = inputs[0][i] * right_gain_;
    }
}

void PanNode::reset() {
    pan_ = 0.0f;
    updateGains();
}

void PanNode::setParameter(int index, float value) {
    if (index == 0) {
        pan_ = std::max(MIN_PAN, std::min(MAX_PAN, value));
        updateGains();
    }
}

float PanNode::getParameter(int index) const {
    if (index == 0) {
        return pan_;
    }
    return 0.0f;
}

void PanNode::setPan(float pan) {
    setParameter(0, pan);
}

float PanNode::getPan() const {
    return pan_;
}

void PanNode::updateGains() {
    left_gain_ = std::cos((pan_ + 1.0f) * M_PI / 4.0f);
    right_gain_ = std::sin((pan_ + 1.0f) * M_PI / 4.0f);
}

// FadeNode implementation
FadeNode::FadeNode(uint32_t id, int channels)
    : AudioProcessorNode(id, RoutingNode::FADE, channels, channels) {
    parameters_.resize(2, 1.0f); // current gain, target gain
    parameter_names_ = {"CurrentGain", "TargetGain"};
}

bool FadeNode::initialize(const RoutingConfig& config) {
    config_ = config;
    current_gain_ = target_gain_ = 1.0f;
    fading_ = false;
    return true;
}

void FadeNode::process(const float** inputs, float** outputs, size_t samples) {
    if (bypassed_) {
        // Pass through
        for (int ch = 0; ch < input_channels_; ++ch) {
            if (inputs[ch] && outputs[ch]) {
                std::copy(inputs[ch], inputs[ch] + samples, outputs[ch]);
            }
        }
        return;
    }

    for (int ch = 0; ch < input_channels_; ++ch) {
        if (!inputs[ch] || !outputs[ch]) {
            continue;
        }

        if (fading_) {
            // Perform fade
            float gain = current_gain_;
            int fade_samples = std::min(static_cast<int>(samples), samples_remaining_);

            for (int i = 0; i < fade_samples; ++i) {
                gain += gain_step_;
                outputs[ch][i] = inputs[ch][i] * gain;
            }

            // Copy remaining samples at target gain
            if (fade_samples < static_cast<int>(samples)) {
                for (int i = fade_samples; i < static_cast<int>(samples); ++i) {
                    outputs[ch][i] = inputs[ch][i] * target_gain_;
                }
            }

            samples_remaining_ -= fade_samples;
            current_gain_ = gain;

            if (samples_remaining_ <= 0) {
                fading_ = false;
                current_gain_ = target_gain_;
            }
        } else {
            // Apply current gain
            for (size_t i = 0; i < samples; ++i) {
                outputs[ch][i] = inputs[ch][i] * current_gain_;
            }
        }
    }
}

void FadeNode::reset() {
    current_gain_ = target_gain_ = 1.0f;
    fading_ = false;
    samples_remaining_ = 0;
}

void FadeNode::setParameter(int index, float value) {
    if (index == 0) {
        current_gain_ = std::max(MIN_GAIN, std::min(MAX_GAIN, value));
    } else if (index == 1) {
        target_gain_ = std::max(MIN_GAIN, std::min(MAX_GAIN, value));
    }
}

float FadeNode::getParameter(int index) const {
    if (index == 0) {
        return current_gain_;
    } else if (index == 1) {
        return target_gain_;
    }
    return 0.0f;
}

void FadeNode::startFade(float target_gain, float duration_seconds) {
    target_gain_ = std::max(MIN_GAIN, std::min(MAX_GAIN, target_gain));
    fade_duration_samples_ = static_cast<int>(duration_seconds * config_.sample_rate);
    samples_remaining_ = fade_duration_samples_;
    gain_step_ = (target_gain_ - current_gain_) / static_cast<float>(fade_duration_samples_);
    fading_ = true;
}

// MatrixMixerNode implementation
MatrixMixerNode::MatrixMixerNode(uint32_t id, int inputs, int outputs)
    : AudioProcessorNode(id, RoutingNode::MATRIX, inputs, outputs) {
    matrix_.resize(inputs, std::vector<float>(outputs, 0.0f));
}

bool MatrixMixerNode::initialize(const RoutingConfig& config) {
    config_ = config;
    clearMatrix();
    return true;
}

void MatrixMixerNode::process(const float** inputs, float** outputs, size_t samples) {
    if (bypassed_) {
        // Pass through (connective matrix)
        for (int ch = 0; ch < std::min(input_channels_, output_channels_); ++ch) {
            if (inputs[ch] && outputs[ch]) {
                std::copy(inputs[ch], inputs[ch] + samples, outputs[ch]);
            }
        }
        return;
    }

    // Clear outputs
    for (int out_ch = 0; out_ch < output_channels_; ++out_ch) {
        if (outputs[out_ch]) {
            std::memset(outputs[out_ch], 0, samples * sizeof(float));
        }
    }

    // Mix according to matrix
    for (int in_ch = 0; in_ch < input_channels_; ++in_ch) {
        if (!inputs[in_ch]) continue;

        for (int out_ch = 0; out_ch < output_channels_; ++out_ch) {
            if (!outputs[out_ch]) continue;

            float gain = matrix_[in_ch][out_ch];
            if (std::abs(gain) < 0.0001f) continue;

            for (size_t i = 0; i < samples; ++i) {
                outputs[out_ch][i] += inputs[in_ch][i] * gain;
            }
        }
    }
}

void MatrixMixerNode::reset() {
    // Nothing to reset for static matrix
}

void MatrixMixerNode::setParameter(int index, float value) {
    int total_params = input_channels_ * output_channels_;
    if (index >= 0 && index < total_params) {
        int input = index / output_channels_;
        int output = index % output_channels_;
        matrix_[input][output] = value;
    }
}

float MatrixMixerNode::getParameter(int index) const {
    int total_params = input_channels_ * output_channels_;
    if (index >= 0 && index < total_params) {
        int input = index / output_channels_;
        int output = index % output_channels_;
        return matrix_[input][output];
    }
    return 0.0f;
}

void MatrixMixerNode::setMatrixGain(int input, int output, float gain) {
    if (input >= 0 && input < input_channels_ && output >= 0 && output < output_channels_) {
        matrix_[input][output] = gain;
    }
}

float MatrixMixerNode::getMatrixGain(int input, int output) const {
    if (input >= 0 && input < input_channels_ && output >= 0 && output < output_channels_) {
        return matrix_[input][output];
    }
    return 0.0f;
}

void MatrixMixerNode::clearMatrix() {
    for (auto& row : matrix_) {
        std::fill(row.begin(), row.end(), 0.0f);
    }
}

// AudioRouter implementation
AudioRouter::AudioRouter() : last_metrics_update_(std::chrono::steady_clock::now()),
                           start_time_(std::chrono::high_resolution_clock::now()) {
}

AudioRouter::~AudioRouter() {
    shutdown();
}

bool AudioRouter::initialize(const RoutingConfig& config) {
    if (initialized_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    config_ = config;

    // Validate configuration
    if (!validateConfiguration()) {
        return false;
    }

    // Allocate buffers
    allocateBuffers();

    // Initialize metrics
    std::memset(&metrics_, 0, sizeof(metrics_));
    last_metrics_update_ = std::chrono::steady_clock::now();
    total_frames_processed_ = 0;

    initialized_ = true;
    return true;
}

void AudioRouter::shutdown() {
    if (!initialized_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Clear nodes and paths
    nodes_.clear();
    paths_.clear();
    process_order_.clear();
    process_order_valid_ = false;

    // Deallocate buffers
    deallocateBuffers();

    // Clear callbacks
    routing_callback_ = nullptr;
    node_callback_ = nullptr;
    metrics_callback_ = nullptr;

    initialized_ = false;
}

bool AudioRouter::processAudioFrame(const float** input_buffers, float** output_buffers, size_t num_samples) {
    if (!initialized_ || !input_buffers || !output_buffers || num_samples == 0) {
        return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::lock_guard<std::mutex> lock(mutex_);

    try {
        // Process node graph
        processNodeGraph(input_buffers, output_buffers, num_samples);

        // Update metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        double process_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        total_frames_processed_++;

        // Check for processing deadline miss
        if (process_time > config_.max_process_time.count() / 1000.0) {
            dropped_frames_++;
        }

        // Update metrics periodically
        if (total_frames_processed_ % METRICS_UPDATE_INTERVAL_FRAMES == 0) {
            updateMetrics();
        }

        return true;
    } catch (const std::exception& e) {
        // Handle processing errors
        buffer_overruns_++;
        return false;
    }
}

bool AudioRouter::addNode(const AudioNode& node) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!validateNode(node)) {
        return false;
    }

    // Check for duplicate ID
    if (nodes_.find(node.id) != nodes_.end()) {
        return false;
    }

    auto node_internal = std::make_unique<NodeInternal>();
    node_internal->info = node;
    node_internal->processor = createProcessor(node.type, node.id, node.input_channels, node.output_channels);

    if (!node_internal->processor || !node_internal->processor->initialize(config_)) {
        return false;
    }

    // Allocate buffers for this node
    node_internal->input_buffers.resize(node.input_channels, nullptr);
    node_internal->output_buffers.resize(node.output_channels, nullptr);
    node_internal->temp_buffer.resize(num_samples * std::max(node.input_channels, node.output_channels), 0.0f);

    nodes_[node.id] = std::move(node_internal);
    process_order_valid_ = false;

    // Notify callback
    if (node_callback_) {
        node_callback_(node);
    }

    return true;
}

bool AudioRouter::removeNode(uint32_t node_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = nodes_.find(node_id);
    if (it == nodes_.end()) {
        return false;
    }

    // Remove all paths connected to this node
    auto path_it = paths_.begin();
    while (path_it != paths_.end()) {
        uint64_t path_id = path_it->first;
        uint32_t source_id = static_cast<uint32_t>(path_id >> 32);
        uint32_t dest_id = static_cast<uint32_t>(path_id & 0xFFFFFFFF);

        if (source_id == node_id || dest_id == node_id) {
            path_it = paths_.erase(path_it);
        } else {
            ++path_it;
        }
    }

    nodes_.erase(it);
    process_order_valid_ = false;

    return true;
}

std::optional<AudioNode> AudioRouter::getNode(uint32_t node_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = nodes_.find(node_id);
    if (it != nodes_.end()) {
        return it->second->info;
    }

    return std::nullopt;
}

bool AudioRouter::addPath(const RoutingPath& path) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!validatePath(path)) {
        return false;
    }

    uint64_t path_id = generatePathId(path.source_node_id, path.destination_node_id);

    // Check for duplicate path
    if (paths_.find(path_id) != paths_.end()) {
        return false;
    }

    auto path_internal = std::make_unique<PathInternal>();
    path_internal->path = path;
    path_internal->current_gain = path.gain;
    path_internal->target_gain = path.gain;
    path_internal->current_delay = path.delay_samples;
    path_internal->target_delay = path.delay_samples;
    path_internal->gain_ramping = false;

    // Create delay line if needed
    if (path.delay_samples > 0) {
        int max_delay = path.delay_samples + 1024; // Add some extra buffer
        path_internal->delay_line = std::make_unique<LockFreeRingBuffer>(max_delay);
    }

    paths_[path_id] = std::move(path_internal);
    process_order_valid_ = false;

    // Notify callback
    if (routing_callback_) {
        routing_callback_(path);
    }

    return true;
}

bool AudioRouter::removePath(uint32_t path_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = paths_.find(path_id);
    if (it == paths_.end()) {
        return false;
    }

    paths_.erase(it);
    process_order_valid_ = false;

    return true;
}

std::optional<RoutingPath> AudioRouter::getPath(uint32_t source_id, uint32_t destination_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    uint64_t path_id = generatePathId(source_id, destination_id);
    auto it = paths_.find(path_id);

    if (it != paths_.end()) {
        return it->second->path;
    }

    return std::nullopt;
}

bool AudioRouter::updatePath(uint32_t path_id, float gain, float pan, bool muted) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = paths_.find(path_id);
    if (it == paths_.end()) {
        return false;
    }

    bool updated = false;

    if (gain >= 0.0f) {
        it->second->target_gain = std::max(MIN_GAIN, std::min(MAX_GAIN, gain));
        if (std::abs(it->second->target_gain - it->second->current_gain) > 0.0001f) {
            it->second->gain_ramping = true;
        }
        it->second->path.gain = gain;
        updated = true;
    }

    if (pan >= -999.0f) { // Special value to indicate no pan change
        it->second->path.pan = std::max(MIN_PAN, std::min(MAX_PAN, pan));
        updated = true;
    }

    if (muted != -1) { // Special value to indicate no mute change
        it->second->path.muted = muted;
        updated = true;
    }

    if (updated && routing_callback_) {
        routing_callback_(it->second->path);
    }

    return updated;
}

bool AudioRouter::setPathEnabled(uint32_t source_id, uint32_t destination_id, bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);

    uint64_t path_id = generatePathId(source_id, destination_id);
    auto it = paths_.find(path_id);

    if (it != paths_.end()) {
        it->second->path.enabled = enabled;
        if (routing_callback_) {
            routing_callback_(it->second->path);
        }
        return true;
    }

    return false;
}

bool AudioRouter::setPathDelay(uint32_t source_id, uint32_t destination_id, int32_t delay_samples) {
    std::lock_guard<std::mutex> lock(mutex_);

    uint64_t path_id = generatePathId(source_id, destination_id);
    auto it = paths_.find(path_id);

    if (it != paths_.end()) {
        it->second->target_delay = delay_samples;
        it->second->path.delay_samples = delay_samples;

        // Create or resize delay line if needed
        if (delay_samples > 0) {
            int max_delay = delay_samples + 1024;
            if (!it->second->delay_line) {
                it->second->delay_line = std::make_unique<LockFreeRingBuffer>(max_delay);
            }
            // Note: In a real implementation, we'd need to handle buffer resizing
        }

        if (routing_callback_) {
            routing_callback_(it->second->path);
        }
        return true;
    }

    return false;
}

uint32_t AudioRouter::createProcessorNode(RoutingNode type, const std::string& name,
                                         int input_channels, int output_channels) {
    AudioNode node;
    node.id = next_node_id_++;
    node.type = type;
    node.name = name;
    node.input_channels = input_channels;
    node.output_channels = output_channels;
    node.is_active = true;
    node.supports_bypass = true;
    node.bypassed = false;
    node.latency_mode = config_.default_latency;
    node.buffer_mode = config_.default_buffer;
    node.priority = RoutingPriority::NORMAL;

    if (addNode(node)) {
        return node.id;
    }

    return 0;
}

bool AudioRouter::connectNodes(uint32_t source_id, uint32_t destination_id,
                              float gain, const std::string& path_name) {
    // Create routing path
    RoutingPath path;
    path.source_node_id = source_id;
    path.destination_node_id = destination_id;
    path.gain = gain;
    path.pan = 0.0f;
    path.muted = false;
    path.enabled = true;
    path.delay_samples = 0;
    path.mode = RoutingMode::PASS_THROUGH;
    path.priority = RoutingPriority::NORMAL;
    path.name = path_name.empty() ? ("Path " + std::to_string(source_id) + " -> " + std::to_string(destination_id)) : path_name;
    path.creation_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    return addPath(path);
}

bool AudioRouter::disconnectNodes(uint32_t source_id, uint32_t destination_id) {
    uint64_t path_id = generatePathId(source_id, destination_id);
    return removePath(path_id);
}

std::vector<RoutingPath> AudioRouter::getActivePaths() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<RoutingPath> active_paths;

    for (const auto& [path_id, path_internal] : paths_) {
        if (path_internal->path.enabled && isPathActive(path_id)) {
            active_paths.push_back(path_internal->path);
        }
    }

    return active_paths;
}

std::vector<AudioNode> AudioRouter::getAllNodes() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<AudioNode> all_nodes;
    all_nodes.reserve(nodes_.size());

    for (const auto& [node_id, node_internal] : nodes_) {
        all_nodes.push_back(node_internal->info);
    }

    return all_nodes;
}

std::vector<AudioNode> AudioRouter::getNodesByType(RoutingNode type) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<AudioNode> matching_nodes;

    for (const auto& [node_id, node_internal] : nodes_) {
        if (node_internal->info.type == type) {
            matching_nodes.push_back(node_internal->info);
        }
    }

    return matching_nodes;
}

std::optional<RoutingPath> AudioRouter::findPathByName(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& [path_id, path_internal] : paths_) {
        if (path_internal->path.name == name) {
            return path_internal->path;
        }
    }

    return std::nullopt;
}

std::optional<AudioNode> AudioRouter::findNodeByName(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& [node_id, node_internal] : nodes_) {
        if (node_internal->info.name == name) {
            return node_internal->info;
        }
    }

    return std::nullopt;
}

bool AudioRouter::setRoutingMatrix(uint32_t node_id, const std::vector<std::vector<float>>& matrix) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = nodes_.find(node_id);
    if (it == nodes_.end()) {
        return false;
    }

    auto* matrix_node = dynamic_cast<MatrixMixerNode*>(it->second->processor.get());
    if (!matrix_node) {
        return false;
    }

    // Validate matrix dimensions
    if (matrix.size() != static_cast<size_t>(matrix_node->getInputChannels())) {
        return false;
    }

    for (const auto& row : matrix) {
        if (row.size() != static_cast<size_t>(matrix_node->getOutputChannels())) {
            return false;
        }
    }

    // Set matrix values
    for (size_t input = 0; input < matrix.size(); ++input) {
        for (size_t output = 0; output < matrix[input].size(); ++output) {
            matrix_node->setMatrixGain(static_cast<int>(input), static_cast<int>(output), matrix[input][output]);
        }
    }

    return true;
}

std::optional<std::vector<std::vector<float>>> AudioRouter::getRoutingMatrix(uint32_t node_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = nodes_.find(node_id);
    if (it == nodes_.end()) {
        return std::nullopt;
    }

    const auto* matrix_node = dynamic_cast<const MatrixMixerNode*>(it->second->processor.get());
    if (!matrix_node) {
        return std::nullopt;
    }

    int inputs = matrix_node->getInputChannels();
    int outputs = matrix_node->getOutputChannels();

    std::vector<std::vector<float>> matrix(inputs, std::vector<float>(outputs));

    for (int input = 0; input < inputs; ++input) {
        for (int output = 0; output < outputs; ++output) {
            matrix[input][output] = matrix_node->getMatrixGain(input, output);
        }
    }

    return matrix;
}

bool AudioRouter::startNodeFade(uint32_t node_id, float target_gain, float duration_seconds) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = nodes_.find(node_id);
    if (it == nodes_.end()) {
        return false;
    }

    auto* gain_node = dynamic_cast<GainNode*>(it->second->processor.get());
    if (!gain_node) {
        return false;
    }

    gain_node->setParameter(1, target_gain); // Set target gain

    // Start fade (simplified - would need proper fade implementation)
    float current_gain = gain_node->getParameter(0);
    int fade_samples = static_cast<int>(duration_seconds * config_.sample_rate);
    float gain_step = (target_gain - current_gain) / fade_samples;

    // This is simplified - real implementation would need proper fade handling
    return true;
}

bool AudioRouter::startCrossfade(uint32_t source_id, uint32_t destination_id, float duration_seconds) {
    // Crossfade implementation would go here
    // This would involve setting up fade paths between two nodes
    return true;
}

RoutingMetrics AudioRouter::getMetrics() const {
    std::lock_guard<std::mutex> metrics_lock(metrics_mutex_);
    return metrics_;
}

void AudioRouter::setRoutingCallback(RoutingCallback callback) {
    routing_callback_ = callback;
}

void AudioRouter::setNodeCallback(NodeCallback callback) {
    node_callback_ = callback;
}

void AudioRouter::setMetricsCallback(MetricsCallback callback) {
    metrics_callback_ = callback;
}

void AudioRouter::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Reset all nodes
    for (auto& [node_id, node_internal] : nodes_) {
        if (node_internal->processor) {
            node_internal->processor->reset();
        }
    }

    // Reset all paths
    for (auto& [path_id, path_internal] : paths_) {
        path_internal->current_gain = path_internal->target_gain;
        path_internal->gain_ramping = false;
        path_internal->current_delay = path_internal->target_delay;
        if (path_internal->delay_line) {
            path_internal->delay_line->clear();
        }
    }

    // Reset metrics
    {
        std::lock_guard<std::mutex> metrics_lock(metrics_mutex_);
        std::memset(&metrics_, 0, sizeof(metrics_));
        total_frames_processed_ = 0;
    }
}

void AudioRouter::clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    nodes_.clear();
    paths_.clear();
    process_order_.clear();
    process_order_valid_ = false;

    deallocateBuffers();
    allocateBuffers();
}

bool AudioRouter::validateConfiguration() const {
    return config_.sample_rate > 0 &&
           config_.buffer_size > 0 &&
           config_.max_channels > 0 &&
           config_.max_nodes > 0 &&
           config_.max_paths > 0 &&
           config_.target_latency_ms >= 0.0 &&
           config_.max_acceptable_latency_ms >= config_.target_latency_ms;
}

bool AudioRouter::optimizeRouting() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Update process order
    updateProcessOrder();

    // Optimize for performance
    optimizeProcessOrder();

    // Check for cycles
    detectCycleDependencies();

    return true;
}

std::string AudioRouter::exportConfiguration() const {
    std::lock_guard<std::mutex> lock(mutex_);

    // JSON export implementation would go here
    return "{}";
}

bool AudioRouter::importConfiguration(const std::string& config_json) {
    // JSON import implementation would go here
    return false;
}

// Private methods
bool AudioRouter::validateNode(const AudioNode& node) const {
    return node.id > 0 &&
           node.input_channels >= 0 &&
           node.output_channels >= 0 &&
           node.max_inputs >= 0 &&
           node.max_outputs >= 0 &&
           (node.input_channels <= config_.max_channels) &&
           (node.output_channels <= config_.max_channels);
}

bool AudioRouter::validatePath(const RoutingPath& path) const {
    return path.source_node_id != 0 &&
           path.destination_node_id != 0 &&
           path.source_node_id != path.destination_node_id &&
           path.gain >= 0.0f &&
           path.pan >= -1.0f && path.pan <= 1.0f;
}

void AudioRouter::updateProcessOrder() {
    process_order_.clear();
    process_order_.reserve(nodes_.size());

    // Simple topological sort
    std::unordered_set<uint32_t> visited;
    std::unordered_set<uint32_t> processing;

    for (const auto& [node_id, node_internal] : nodes_) {
        if (visited.find(node_id) == visited.end()) {
            topologicalSortNode(node_id, visited, processing, process_order_);
        }
    }

    process_order_valid_ = true;
}

void AudioRouter::processNodeGraph(const float** inputs, float** outputs, size_t samples) {
    // Update process order if needed
    if (!process_order_valid_) {
        updateProcessOrder();
    }

    // Set up initial input buffers
    for (int ch = 0; ch < config_.max_channels; ++ch) {
        if (ch < static_cast<int>(input_pointers_.size()) && inputs[ch]) {
            process_buffers_[0][ch] = inputs[ch];
        } else {
            process_buffers_[0][ch] = nullptr;
        }
    }

    // Process nodes in order
    for (auto* node_internal : process_order_) {
        if (!node_internal || !node_internal->processor || !node_internal->info.is_active) {
            continue;
        }

        // Set up input pointers for this node
        for (int ch = 0; ch < node_internal->info.input_channels; ++ch) {
            node_internal->input_buffers[ch] = nullptr;
        }

        for (int ch = 0; ch < node_internal->info.output_channels; ++ch) {
            node_internal->output_buffers[ch] = process_buffers_[0][ch]; // Simplified
        }

        // Process node
        node_internal->processor->process(
            node_internal->input_buffers.data(),
            node_internal->output_buffers.data(),
            samples
        );
    }

    // Apply routing paths
    for (auto& [path_id, path_internal] : paths_) {
        if (!path_internal->path.enabled || !isPathActive(path_id)) {
            continue;
        }

        // Get source and destination nodes
        auto source_it = nodes_.find(path_internal->path.source_node_id);
        auto dest_it = nodes_.find(path_internal->path.destination_node_id);

        if (source_it == nodes_.end() || dest_it == nodes_.end()) {
            continue;
        }

        // Process path (simplified)
        // Real implementation would connect actual audio buffers
        path_internal->process_count++;
        path_internal->last_process = std::chrono::high_resolution_clock::now();
    }

    // Copy to output buffers
    for (int ch = 0; ch < config_.max_channels && ch < static_cast<int>(output_pointers_.size()); ++ch) {
        if (outputs[ch] && process_buffers_[0][ch]) {
            std::copy(process_buffers_[0][ch], process_buffers_[0][ch] + samples, outputs[ch]);
        }
    }
}

void AudioRouter::updateMetrics() {
    std::lock_guard<std::mutex> metrics_lock(metrics_mutex_);

    auto now = std::chrono::steady_clock::now();

    metrics_.total_paths = paths_.size();
    metrics_.active_nodes = 0;
    metrics_.active_paths = 0;

    for (const auto& [node_id, node_internal] : nodes_) {
        if (node_internal->info.is_active) {
            metrics_.active_nodes++;
        }
    }

    for (const auto& [path_id, path_internal] : paths_) {
        if (path_internal->path.enabled && isPathActive(path_id)) {
            metrics_.active_paths++;
        }
    }

    metrics_.buffer_underruns = buffer_underruns_;
    metrics_.buffer_overruns = buffer_overruns_;
    metrics_.xruns_count = metrics_.buffer_underruns + metrics_.buffer_overruns;
    metrics_.dropped_frames = dropped_frames_;
    metrics_.last_update = now;

    // Calculate CPU usage (simplified)
    auto elapsed = std::chrono::duration<double>(now - start_time_).count();
    if (elapsed > 0.0) {
        double expected_time = (static_cast<double>(total_frames_processed_) * config_.buffer_size) / config_.sample_rate;
        metrics_.cpu_usage_percent = (expected_time / elapsed) * 100.0;
        metrics_.real_time_stable = metrics_.cpu_usage_percent < 95.0;
    }

    if (metrics_callback_) {
        metrics_callback_(metrics_);
    }
}

void AudioRouter::updatePathGains(float* buffer, size_t samples, PathInternal& path) {
    if (!path.gain_ramping) {
        return;
    }

    float gain_diff = path.target_gain - path.current_gain;
    if (std::abs(gain_diff) < 0.0001f) {
        path.current_gain = path.target_gain;
        path.gain_ramping = false;
        return;
    }

    float gain_step = gain_diff / static_cast<float>(samples);
    for (size_t i = 0; i < samples; ++i) {
        path.current_gain += gain_step;
        buffer[i] *= path.current_gain;
    }
}

void AudioRouter::updatePathDelay(const float* input, float* output, size_t samples, PathInternal& path) {
    if (!path.delay_line || path.delay_samples == 0) {
        if (input && output) {
            std::copy(input, input + samples, output);
        }
        return;
    }

    if (!input || !output) {
        return;
    }

    // Write input to delay line
    path.delay_line->write(input, samples);

    // Read delayed output
    if (!path.delay_line->read(output, samples)) {
        // Not enough delayed data available
        std::memset(output, 0, samples * sizeof(float));
    }
}

void AudioRouter::mixPathAudio(const float* input, float* output, size_t samples, PathInternal& path) {
    if (!input || !output || path.path.muted) {
        return;
    }

    // Apply gain
    float effective_gain = path.current_gain;
    if (path.gain_ramping) {
        updatePathGains(const_cast<float*>(input), samples, path);
        effective_gain = path.current_gain;
    }

    // Apply pan (simplified - assumes stereo)
    if (path.path.pan != 0.0f) {
        float left_gain = std::cos((path.path.pan + 1.0f) * M_PI / 4.0f);
        float right_gain = std::sin((path.path.pan + 1.0f) * M_PI / 4.0f);

        for (size_t i = 0; i < samples; i += 2) {
            output[i] += input[i] * effective_gain * left_gain;
            if (i + 1 < samples) {
                output[i + 1] += input[i + 1] * effective_gain * right_gain;
            }
        }
    } else {
        for (size_t i = 0; i < samples; ++i) {
            output[i] += input[i] * effective_gain;
        }
    }
}

uint64_t AudioRouter::generatePathId(uint32_t source_id, uint32_t destination_id) const {
    return (static_cast<uint64_t>(source_id) << 32) | static_cast<uint64_t>(destination_id);
}

std::unique_ptr<AudioProcessorNode> AudioRouter::createProcessor(RoutingNode type, uint32_t id,
                                                               int input_channels, int output_channels) {
    switch (type) {
        case RoutingNode::GAIN:
            return std::make_unique<GainNode>(id, input_channels);
        case RoutingNode::MIXER:
            return std::make_unique<MixerNode>(id, input_channels, output_channels);
        case RoutingNode::DELAY:
            return std::make_unique<DelayNode>(id, input_channels, config_.sample_rate * MAX_DELAY_SECONDS);
        case RoutingNode::PAN:
            return std::make_unique<PanNode>(id, input_channels, output_channels);
        case RoutingNode::FADE:
            return std::make_unique<FadeNode>(id, input_channels);
        case RoutingNode::MATRIX:
            return std::make_unique<MatrixMixerNode>(id, input_channels, output_channels);
        default:
            return nullptr; // Unsupported type
    }
}

void AudioRouter::optimizeProcessOrder() {
    // Process order optimization would go here
    // This would prioritize critical paths, group related nodes, etc.
}

void AudioRouter::detectCycleDependencies() {
    // Cycle detection would go here
    // This would check for feedback loops and handle them appropriately
}

bool AudioRouter::isPathActive(uint64_t path_id) const {
    auto it = paths_.find(path_id);
    if (it == paths_.end()) {
        return false;
    }

    uint32_t source_id = it->second->path.source_node_id;
    uint32_t dest_id = it->second->path.destination_node_id;

    auto source_it = nodes_.find(source_id);
    auto dest_it = nodes_.find(dest_id);

    return (source_it != nodes_.end() && source_it->second->info.is_active) &&
           (dest_it != nodes_.end() && dest_it->second->info.is_active);
}

void AudioRouter::clearNodeBuffers() {
    for (auto& buffer : process_buffers_) {
        std::fill(buffer.begin(), buffer.end(), 0.0f);
    }
}

void AudioRouter::allocateBuffers() {
    int max_channels = config_.max_channels;
    size_t buffer_size = config_.buffer_size;

    process_buffers_.resize(1, std::vector<float>(max_channels * buffer_size, 0.0f));
    input_pointers_.resize(max_channels, nullptr);
    output_pointers_.resize(max_channels, nullptr);

    // Initialize pointers to buffer memory
    for (int ch = 0; ch < max_channels; ++ch) {
        process_buffers_[0][ch] = nullptr; // Will be set during processing
        input_pointers_[ch] = nullptr;
        output_pointers_[ch] = nullptr;
    }
}

void AudioRouter::deallocateBuffers() {
    process_buffers_.clear();
    input_pointers_.clear();
    output_pointers_.clear();
}

// AudioRouterFactory implementation
std::unique_ptr<AudioRouter> AudioRouterFactory::createMixingConsole(int input_channels, int output_channels,
                                                                  int sample_rate, int buffer_size) {
    auto config = createOptimalConfig(std::max(input_channels, output_channels), sample_rate, buffer_size);

    auto router = std::make_unique<AudioRouter>();
    if (!router->initialize(config)) {
        return nullptr;
    }

    // Create input gain nodes
    for (int i = 0; i < input_channels; ++i) {
        uint32_t gain_id = router->createProcessorNode(RoutingNode::GAIN, "Input " + std::to_string(i) + " Gain", 1, 1);
        if (gain_id == 0) return nullptr;
    }

    // Create main mixer
    uint32_t mixer_id = router->createProcessorNode(RoutingNode::MIXER, "Main Mixer", input_channels, output_channels);
    if (mixer_id == 0) return nullptr;

    // Create output gain nodes
    for (int i = 0; i < output_channels; ++i) {
        uint32_t gain_id = router->createProcessorNode(RoutingNode::GAIN, "Output " + std::to_string(i) + " Gain", 1, 1);
        if (gain_id == 0) return nullptr;
    }

    return router;
}

std::unique_ptr<AudioRouter> AudioRouterFactory::createDAWRouting(int tracks, int buses,
                                                               int sample_rate, int buffer_size) {
    auto config = createOptimalConfig((tracks + buses) * 2, sample_rate, buffer_size, LatencyMode::BALANCED);

    auto router = std::make_unique<AudioRouter>();
    if (!router->initialize(config)) {
        return nullptr;
    }

    // Create track channel strips
    for (int i = 0; i < tracks; ++i) {
        uint32_t gain_id = router->createProcessorNode(RoutingNode::GAIN, "Track " + std::to_string(i) + " Gain", 2, 2);
        uint32_t pan_id = router->createProcessorNode(RoutingNode::PAN, "Track " + std::to_string(i) + " Pan", 2, 2);
        if (gain_id == 0 || pan_id == 0) return nullptr;
    }

    // Create mixer buses
    for (int i = 0; i < buses; ++i) {
        uint32_t bus_id = router->createProcessorNode(RoutingNode::MIXER, "Bus " + std::to_string(i), tracks * 2, 2);
        if (bus_id == 0) return nullptr;
    }

    // Create master mixer
    uint32_t master_id = router->createProcessorNode(RoutingNode::MIXER, "Master", buses * 2, 2);
    if (master_id == 0) return nullptr;

    return router;
}

std::unique_ptr<AudioRouter> AudioRouterFactory::createLiveSoundMixer(int inputs, int outputs, int matrix_size,
                                                                    int sample_rate, int buffer_size) {
    auto config = createOptimalConfig(std::max(inputs, outputs), sample_rate, buffer_size, LatencyMode::LOWEST);

    auto router = std::make_unique<AudioRouter>();
    if (!router->initialize(config)) {
        return nullptr;
    }

    // Create input gains
    for (int i = 0; i < inputs; ++i) {
        uint32_t gain_id = router->createProcessorNode(RoutingNode::GAIN, "Input " + std::to_string(i) + " Gain", 1, 1);
        if (gain_id == 0) return nullptr;
    }

    // Create matrix mixer
    uint32_t matrix_id = router->createProcessorNode(RoutingNode::MATRIX, "Matrix Mixer", inputs, outputs);
    if (matrix_id == 0) return nullptr;

    // Create output gains
    for (int i = 0; i < outputs; ++i) {
        uint32_t gain_id = router->createProcessorNode(RoutingNode::GAIN, "Output " + std::to_string(i) + " Gain", 1, 1);
        if (gain_id == 0) return nullptr;
    }

    return router;
}

std::unique_ptr<AudioRouter> AudioRouterFactory::createBroadcastMixer(int channels, int aux_sends,
                                                                    int sample_rate, int buffer_size) {
    auto config = createOptimalConfig(channels + aux_sends, sample_rate, buffer_size);

    auto router = std::make_unique<AudioRouter>();
    if (!router->initialize(config)) {
        return nullptr;
    }

    // Create main channels
    for (int i = 0; i < channels; ++i) {
        uint32_t gain_id = router->createProcessorNode(RoutingNode::GAIN, "Channel " + std::to_string(i) + " Gain", 1, 1);
        uint32_t fader_id = router->createProcessorNode(RoutingNode::FADE, "Channel " + std::to_string(i) + " Fader", 1, 1);
        if (gain_id == 0 || fader_id == 0) return nullptr;
    }

    // Create aux sends
    for (int i = 0; i < aux_sends; ++i) {
        uint32_t aux_id = router->createProcessorNode(RoutingNode::MIXER, "Aux " + std::to_string(i), channels, 1);
        if (aux_id == 0) return nullptr;
    }

    // Create main mix bus
    uint32_t main_id = router->createProcessorNode(RoutingNode::MIXER, "Main Mix", channels, 2);
    if (main_id == 0) return nullptr;

    return router;
}

std::unique_ptr<AudioRouter> AudioRouterFactory::createMinimalLatencyRouter(int channels, double target_latency_ms,
                                                                           int sample_rate) {
    auto config = createLowLatencyConfig(sample_rate, channels, target_latency_ms);

    auto router = std::make_unique<AudioRouter>();
    if (!router->initialize(config)) {
        return nullptr;
    }

    // Create simple gain nodes for minimal latency
    for (int i = 0; i < channels; ++i) {
        uint32_t gain_id = router->createProcessorNode(RoutingNode::GAIN, "Channel " + std::to_string(i) + " Gain", 1, 1);
        if (gain_id == 0) return nullptr;
    }

    return router;
}

RoutingConfig AudioRouterFactory::createOptimalConfig(int channels, int sample_rate, int buffer_size,
                                                     LatencyMode latency) {
    RoutingConfig config;
    config.sample_rate = sample_rate;
    config.buffer_size = buffer_size;
    config.max_channels = channels;
    config.max_nodes = channels * 2 + 16; // Account for processing nodes
    config.max_paths = channels * channels + 64;
    config.default_latency = latency;
    config.default_buffer = BufferMode::LOCK_FREE;
    config.target_latency_ms = audio_routing_utils::calculateLatency(buffer_size, sample_rate);
    config.max_acceptable_latency_ms = config.target_latency_ms * 2.0;
    config.enable_deterministic = true;
    config.enable_zero_copy = false;
    config.enable_profiling = false;

    // Set thread priority based on latency mode
    switch (latency) {
        case LatencyMode::MINIMUM:
        case LatencyMode::LOW:
            config.thread_priority = 5; // High priority
            break;
        case LatencyMode::BALANCED:
            config.thread_priority = 0; // Normal priority
            break;
        case LatencyMode::SAFE:
        case LatencyMode::CUSTOM:
            config.thread_priority = -2; // Low priority
            break;
    }

    return config;
}

// Utility functions implementation
namespace audio_routing_utils {

uint64_t calculatePathHash(uint32_t source, uint32_t destination) {
    return (static_cast<uint64_t>(source) << 32) | static_cast<uint64_t>(destination);
}

bool isNodeConnected(uint32_t node_id, const std::vector<RoutingPath>& paths, bool as_source) {
    for (const auto& path : paths) {
        if (as_source && path.source_node_id == node_id) {
            return true;
        } else if (!as_source && path.destination_node_id == node_id) {
            return true;
        }
    }
    return false;
}

std::vector<uint32_t> findConnectedNodes(uint32_t node_id, const std::vector<RoutingPath>& paths, bool as_source) {
    std::vector<uint32_t> connected;

    for (const auto& path : paths) {
        uint32_t connected_id = as_source ? path.destination_node_id : path.source_node_id;
        uint32_t check_id = as_source ? path.source_node_id : path.destination_node_id;

        if (check_id == node_id) {
            connected.push_back(connected_id);
        }
    }

    return connected;
}

bool hasCycle(uint32_t start_node, const std::vector<RoutingPath>& paths) {
    std::unordered_set<uint32_t> visited;
    std::unordered_set<uint32_t> recursion_stack;

    return hasCycleUtil(start_node, paths, visited, recursion_stack);
}

std::vector<uint32_t> topologicalSort(const std::vector<RoutingPath>& paths, const std::vector<uint32_t>& nodes) {
    std::unordered_map<uint32_t, std::vector<uint32_t>> graph;
    std::unordered_map<uint32_t, int> in_degree;

    // Build graph
    for (uint32_t node : nodes) {
        graph[node] = std::vector<uint32_t>();
        in_degree[node] = 0;
    }

    for (const auto& path : paths) {
        graph[path.source_node_id].push_back(path.destination_node_id);
        in_degree[path.destination_node_id]++;
    }

    // Kahn's algorithm for topological sort
    std::queue<uint32_t> queue;
    std::vector<uint32_t> result;

    for (const auto& [node, degree] : in_degree) {
        if (degree == 0) {
            queue.push(node);
        }
    }

    while (!queue.empty()) {
        uint32_t current = queue.front();
        queue.pop();
        result.push_back(current);

        for (uint32_t neighbor : graph[current]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                queue.push(neighbor);
            }
        }
    }

    return result;
}

void applyGain(float* buffer, size_t samples, float gain) {
    if (!buffer || samples == 0) {
        return;
    }

    for (size_t i = 0; i < samples; ++i) {
        buffer[i] *= gain;
    }
}

void applyPan(const float* input, float* output, size_t samples, float pan, int channels) {
    if (!input || !output || samples == 0 || channels < 2) {
        return;
    }

    float left_gain = std::cos((pan + 1.0f) * M_PI / 4.0f);
    float right_gain = std::sin((pan + 1.0f) * M_PI / 4.0f);

    for (size_t i = 0; i < samples; ++i) {
        output[i * 2] = input[i] * left_gain;
        output[i * 2 + 1] = input[i] * right_gain;
    }
}

void crossfade(const float* input1, const float* input2, float* output, size_t samples, float mix) {
    if (!input1 || !input2 || !output || samples == 0) {
        return;
    }

    mix = std::max(0.0f, std::min(1.0f, mix));
    float gain1 = 1.0f - mix;
    float gain2 = mix;

    for (size_t i = 0; i < samples; ++i) {
        output[i] = input1[i] * gain1 + input2[i] * gain2;
    }
}

void linearFade(float* buffer, size_t samples, float start_gain, float end_gain) {
    if (!buffer || samples == 0) {
        return;
    }

    float gain_step = (end_gain - start_gain) / static_cast<float>(samples);

    for (size_t i = 0; i < samples; ++i) {
        float gain = start_gain + gain_step * i;
        buffer[i] *= gain;
    }
}

float calculateRMS(const float* buffer, size_t samples) {
    if (!buffer || samples == 0) {
        return 0.0f;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < samples; ++i) {
        sum += buffer[i] * buffer[i];
    }

    return std::sqrt(sum / static_cast<float>(samples));
}

float calculatePeak(const float* buffer, size_t samples) {
    if (!buffer || samples == 0) {
        return 0.0f;
    }

    float peak = 0.0f;
    for (size_t i = 0; i < samples; ++i) {
        peak = std::max(peak, std::abs(buffer[i]));
    }

    return peak;
}

double calculateLatency(int buffer_size, int sample_rate) {
    return (static_cast<double>(buffer_size) / sample_rate) * 1000.0;
}

int calculateOptimalBufferSize(int sample_rate, double target_latency_ms) {
    int buffer_size = static_cast<int>((target_latency_ms / 1000.0) * sample_rate);

    // Round to nearest power of 2
    int power_of_2 = 1;
    while (power_of_2 < buffer_size) {
        power_of_2 *= 2;
    }

    return std::max(32, std::min(8192, power_of_2));
}

size_t calculateMemoryUsage(int channels, int buffer_size, int nodes, int paths) {
    size_t memory_usage = 0;

    // Audio buffers
    memory_usage += channels * buffer_size * sizeof(float);

    // Node storage
    memory_usage += nodes * sizeof(AudioNode);
    memory_usage += nodes * sizeof(AudioProcessorNode*);

    // Path storage
    memory_usage += paths * sizeof(RoutingPath);

    // Process buffers
    memory_usage += nodes * channels * buffer_size * sizeof(float);

    return memory_usage / (1024 * 1024); // Return in MB
}

bool isLatencyAcceptable(double actual_latency_ms, double target_latency_ms) {
    return actual_latency_ms <= target_latency_ms * 1.5; // Allow 50% overhead
}

RoutingConfig createLowLatencyConfig(int sample_rate, int target_channels, double target_latency_ms) {
    RoutingConfig config;
    config.sample_rate = sample_rate;
    config.buffer_size = calculateOptimalBufferSize(sample_rate, target_latency_ms);
    config.max_channels = target_channels;
    config.max_nodes = target_channels * 4;
    config.max_paths = target_channels * target_channels;
    config.default_latency = LatencyMode::LOWEST;
    config.default_buffer = BufferMode::LOCK_FREE;
    config.target_latency_ms = target_latency_ms;
    config.max_acceptable_latency_ms = target_latency_ms * 1.2;
    config.enable_deterministic = true;
    config.enable_zero_copy = true;
    config.thread_priority = 10; // Maximum priority
    config.max_process_time = std::chrono::microseconds(static_cast<int>(target_latency_ms * 500)); // Half of target

    return config;
}

RoutingConfig createBalancedConfig(int sample_rate, int target_channels) {
    RoutingConfig config;
    config.sample_rate = sample_rate;
    config.buffer_size = 256;
    config.max_channels = target_channels;
    config.max_nodes = target_channels * 2;
    config.max_paths = target_channels * target_channels / 2;
    config.default_latency = LatencyMode::BALANCED;
    config.default_buffer = BufferMode::LOCK_FREE;
    config.target_latency_ms = 5.0;
    config.max_acceptable_latency_ms = 10.0;
    config.enable_deterministic = true;
    config.enable_zero_copy = false;
    config.thread_priority = 0;
    config.max_process_time = std::chrono::microseconds(2500);

    return config;
}

RoutingConfig createSafeConfig(int sample_rate, int target_channels) {
    RoutingConfig config;
    config.sample_rate = sample_rate;
    config.buffer_size = 1024;
    config.max_channels = target_channels;
    config.max_nodes = target_channels;
    config.max_paths = target_channels;
    config.default_latency = LatencyMode::SAFE;
    config.default_buffer = BufferMode::ATOMIC;
    config.target_latency_ms = 20.0;
    config.max_acceptable_latency_ms = 50.0;
    config.enable_deterministic = false;
    config.enable_zero_copy = false;
    config.thread_priority = -5;
    config.max_process_time = std::chrono::microseconds(10000);

    return config;
}

// Helper function for cycle detection (recursive)
bool hasCycleUtil(uint32_t node, const std::vector<RoutingPath>& paths,
                  std::unordered_set<uint32_t>& visited,
                  std::unordered_set<uint32_t>& recursion_stack) {
    visited.insert(node);
    recursion_stack.insert(node);

    for (uint32_t neighbor : findConnectedNodes(node, paths, true)) {
        if (recursion_stack.find(neighbor) != recursion_stack.end()) {
            return true;
        }

        if (visited.find(neighbor) == visited.end() &&
            hasCycleUtil(neighbor, paths, visited, recursion_stack)) {
            return true;
        }
    }

    recursion_stack.erase(node);
    return false;
}

} // namespace audio_routing_utils

} // namespace audio
} // namespace core
} // namespace vortex