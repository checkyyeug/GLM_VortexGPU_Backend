#include "core/audio/multi_channel_engine.hpp"
#include "core/audio/audio_buffer.hpp"
#include "core/dsp/audio_math.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>
#include <random>

#ifdef _WIN32
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <dsound.h>
#endif

#ifdef __APPLE__
#include <CoreAudio/CoreAudio.h>
#include <AudioUnit/AudioUnit.h>
#endif

#ifdef __linux__
#include <alsa/asoundlib.h>
#include <jack/jack.h>
#endif

namespace vortex {
namespace core {
namespace audio {

// Constants
constexpr int DEFAULT_SAMPLE_RATE = 48000;
constexpr int DEFAULT_BUFFER_SIZE = 512;
constexpr float MAX_GAIN = 100.0f;
constexpr float MIN_GAIN = 0.0f;
constexpr float GAIN_SMOOTHING_TIME_MS = 10.0f;
constexpr float PAN_SMOOTHING_TIME_MS = 5.0f;
constexpr int METRICS_UPDATE_INTERVAL_MS = 1000;
constexpr float XRUN_THRESHOLD_MS = 2.0f;
constexpr int MAX_XRUNS_PER_SECOND = 10;

MultiChannelEngine::MultiChannelEngine()
    : last_metrics_update_(std::chrono::steady_clock::now()) {

    // Initialize processing components
    spatial_processor_ = std::make_unique<SpatialAudioProcessor>();
    spectrum_analyzer_ = std::make_unique<SpectrumAnalyzer>();
    waveform_processor_ = std::make_unique<WaveformProcessor>();
    vu_processor_ = std::make_unique<VUMeterProcessor>();

    // Initialize session with defaults
    session_.sample_rate = DEFAULT_SAMPLE_RATE;
    session_.buffer_size = DEFAULT_BUFFER_SIZE;
    session_.channels = 2;
    session_.layout = AudioChannelLayout::STEREO;
    session_.bit_depth = AudioBitDepth::FLOAT32;
    session_.sync_mode = AudioSyncMode::NONE;
    session_.start_time = std::chrono::steady_clock::now();

    // Initialize routing matrix
    routing_matrix_.input_channels = 0;
    routing_matrix_.output_channels = 0;
    routing_matrix_.matrix.clear();
}

MultiChannelEngine::~MultiChannelEngine() {
    shutdown();
}

bool MultiChannelEngine::initialize(const AudioSession& session) {
    if (initialized_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(session_mutex_);
    session_ = session;

    // Validate session configuration
    if (!validateConfiguration()) {
        return false;
    }

    // Initialize channel layout
    initializeChannelLayout();

    // Initialize devices
    if (!initializeDevices()) {
        return false;
    }

    // Initialize audio drivers
    if (!initializeAudioDrivers()) {
        return false;
    }

    // Initialize processing components
    if (!spatial_processor_->initialize(session_.sample_rate, session_.buffer_size, session_.channels)) {
        return false;
    }

    if (!spectrum_analyzer_->initialize(session_.sample_rate, session_.buffer_size, 2048,
                                       SpectrumAnalyzer::WindowType::Hanning)) {
        return false;
    }

    if (!waveform_processor_->initialize(session_.sample_rate, session_.buffer_size)) {
        return false;
    }

    if (!vu_processor_->initialize(session_.sample_rate, session_.buffer_size, session_.channels)) {
        return false;
    }

    // Initialize buffers
    input_buffer_.resize(session_.buffer_size * session_.channels, 0.0f);
    output_buffer_.resize(session_.buffer_size * session_.channels, 0.0f);
    mix_buffer_.resize(session_.buffer_size * session_.channels, 0.0f);
    temp_buffer_.resize(session_.buffer_size * session_.channels, 0.0f);

    // Initialize peak levels tracking
    peak_levels_.resize(session_.channels, 0.0f);

    // Initialize metrics
    std::memset(&metrics_, 0, sizeof(metrics_));
    metrics_.input_levels.resize(session_.channels, 0.0f);
    metrics_.output_levels.resize(session_.channels, 0.0f);
    last_metrics_update_ = std::chrono::steady_clock::now();
    xruns_count_ = 0;

    initialized_ = true;
    return true;
}

void MultiChannelEngine::shutdown() {
    if (!initialized_) {
        return;
    }

    // Stop audio processing
    stop();

    // Shutdown audio drivers
    shutdownAudioDrivers();

    // Stop recording
    stopRecording();

    // Clear all state
    {
        std::lock_guard<std::mutex> lock(devices_mutex_);
        active_devices_.clear();
    }

    {
        std::lock_guard<std::mutex> lock(channels_mutex_);
        channels_.clear();
    }

    {
        std::lock_guard<std::mutex> lock(buses_mutex_);
        buses_.clear();
    }

    // Shutdown processing components
    if (spatial_processor_) {
        spatial_processor_->shutdown();
    }

    // Clear callbacks
    device_callback_ = nullptr;
    session_callback_ = nullptr;
    metrics_callback_ = nullptr;
    level_callback_ = nullptr;

    initialized_ = false;
}

bool MultiChannelEngine::start() {
    if (!initialized_ || running_) {
        return false;
    }

    // Start all devices
    if (!startDevice(master_device_id_)) {
        return false;
    }

    for (uint32_t device_id : input_device_ids_) {
        if (!startDevice(device_id)) {
            return false;
        }
    }

    for (uint32_t device_id : output_device_ids_) {
        if (device_id != master_device_id_) {
            if (!startDevice(device_id)) {
                return false;
            }
        }
    }

    // Start recording if enabled
    startRecording();

    // Update session state
    session_.start_time = std::chrono::steady_clock::now();
    session_.processed_samples = 0;
    start_time_ = std::chrono::high_resolution_clock::now();

    running_ = true;
    processing_active_ = true;
    processed_frames_ = 0;

    // Notify callbacks
    if (session_callback_) {
        session_callback_(session_);
    }

    return true;
}

void MultiChannelEngine::stop() {
    if (!running_) {
        return;
    }

    running_ = false;
    processing_active_ = false;

    // Stop all devices
    stopDevice(master_device_id_);

    for (uint32_t device_id : input_device_ids_) {
        stopDevice(device_id);
    }

    for (uint32_t device_id : output_device_ids_) {
        if (device_id != master_device_id_) {
            stopDevice(device_id);
        }
    }

    // Stop recording
    stopRecording();

    // Update metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.processed_frames = processed_frames_;
    }
}

bool MultiChannelEngine::processAudioCallback(const float* input_buffer, float* output_buffer, size_t num_samples) {
    if (!initialized_ || !running_ || !input_buffer || !output_buffer || num_samples == 0) {
        return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Clear output buffer
    std::memset(output_buffer, 0, num_samples * session_.channels * sizeof(float));

    // Copy input to internal buffer
    std::copy(input_buffer, input_buffer + num_samples * session_.channels, input_buffer_.begin());

    // Process audio channels
    processChannels(input_buffer_.data(), output_buffer, num_samples);

    // Process audio buses
    processBuses(output_buffer, num_samples);

    // Apply routing matrix
    applyRouting(input_buffer_.data(), output_buffer, num_samples);

    // Apply dithering if needed
    if (session_.enable_dithering && session_.bit_depth != AudioBitDepth::FLOAT32 &&
        session_.bit_depth != AudioBitDepth::FLOAT64) {
        applyDithering(output_buffer, num_samples);
    }

    // Update audio processors
    spectrum_analyzer_->processAudio(output_buffer, num_samples);
    waveform_processor_->processAudio(output_buffer, num_samples);
    vu_processor_->processAudio(output_buffer, num_samples);

    // Update channel levels
    updateChannelLevels(output_buffer, num_samples);

    // Update processing metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    double processing_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.processed_frames++;
        metrics_.average_latency_ms = (metrics_.average_latency_ms * 0.95) + (processing_time * 0.05);
        metrics_.peak_latency_ms = std::max(metrics_.peak_latency_ms, processing_time);

        // Update metrics every second
        auto now = std::chrono::high_resolution_clock::now();
        auto time_since_update = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_metrics_update_).count();

        if (time_since_update >= METRICS_UPDATE_INTERVAL_MS) {
            updateMetrics();
            last_metrics_update_ = now;
        }

        // Check for XRUNs
        if (processing_time > XRUN_THRESHOLD_MS) {
            xruns_count_++;
            metrics_.xruns_count = xruns_count_;
        }
    }

    processed_frames_++;
    session_.processed_samples += num_samples;

    return true;
}

std::vector<AudioDevice> MultiChannelEngine::scanDevices() {
    AudioDeviceManager device_manager;
    if (!device_manager.initialize()) {
        return {};
    }

    std::vector<AudioDevice> devices = device_manager.scanDevices();

    // Update available devices list
    {
        std::lock_guard<std::mutex> lock(devices_mutex_);
        available_devices_.clear();
        for (const auto& device : devices) {
            available_devices_[device.id] = device;
        }
    }

    return devices;
}

std::vector<AudioDevice> MultiChannelEngine::getAvailableDevices() const {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    std::vector<AudioDevice> devices;
    devices.reserve(available_devices_.size());

    for (const auto& [id, device] : available_devices_) {
        devices.push_back(device);
    }

    return devices;
}

std::optional<AudioDevice> MultiChannelEngine::getDevice(uint32_t device_id) const {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    auto it = available_devices_.find(device_id);
    if (it != available_devices_.end()) {
        return it->second;
    }

    return std::nullopt;
}

bool MultiChannelEngine::setMasterDevice(uint32_t device_id) {
    auto device = getDevice(device_id);
    if (!device || device->type != AudioDeviceType::OUTPUT) {
        return false;
    }

    std::lock_guard<std::mutex> lock(devices_mutex_);

    // Stop current master device if running
    if (running_ && master_device_id_ != 0 && master_device_id_ != device_id) {
        performDeviceCrossfade(master_device_id_, device_id);
    }

    master_device_id_ = device_id;

    // Update session
    session_.master_device_id = device_id;

    // Notify callback
    if (device_callback_ && device) {
        device_callback_(*device);
    }

    return true;
}

bool MultiChannelEngine::addInputDevice(uint32_t device_id) {
    auto device = getDevice(device_id);
    if (!device || (device->type != AudioDeviceType::INPUT && device->type != AudioDeviceType::INPUT_OUTPUT)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(devices_mutex_);

    // Check if device already added
    if (std::find(input_device_ids_.begin(), input_device_ids_.end(), device_id) != input_device_ids_.end()) {
        return false;
    }

    input_device_ids_.push_back(device_id);

    // Start device if engine is running
    if (running_) {
        startDevice(device_id);
    }

    // Notify callback
    if (device_callback_ && device) {
        device_callback_(*device);
    }

    return true;
}

bool MultiChannelEngine::addOutputDevice(uint32_t device_id) {
    auto device = getDevice(device_id);
    if (!device || (device->type != AudioDeviceType::OUTPUT && device->type != AudioDeviceType::INPUT_OUTPUT)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(devices_mutex_);

    // Check if device already added
    if (std::find(output_device_ids_.begin(), output_device_ids_.end(), device_id) != output_device_ids_.end()) {
        return false;
    }

    output_device_ids_.push_back(device_id);

    // Start device if engine is running
    if (running_) {
        startDevice(device_id);
    }

    // Notify callback
    if (device_callback_ && device) {
        device_callback_(*device);
    }

    return true;
}

bool MultiChannelEngine::removeDevice(uint32_t device_id) {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    bool device_removed = false;

    // Remove from input devices
    auto input_it = std::find(input_device_ids_.begin(), input_device_ids_.end(), device_id);
    if (input_it != input_device_ids_.end()) {
        input_device_ids_.erase(input_it);
        device_removed = true;
    }

    // Remove from output devices
    auto output_it = std::find(output_device_ids_.begin(), output_device_ids_.end(), device_id);
    if (output_it != output_device_ids_.end()) {
        output_device_ids_.erase(output_it);
        device_removed = true;
    }

    // Stop device if it was the master device
    if (master_device_id_ == device_id) {
        stopDevice(device_id);
        master_device_id_ = 0;
        device_removed = true;
    }

    // Stop device if engine is running
    if (running_) {
        stopDevice(device_id);
    }

    // Remove from active devices
    auto active_it = active_devices_.find(device_id);
    if (active_it != active_devices_.end()) {
        active_devices_.erase(active_it);
        device_removed = true;
    }

    // Notify callback
    if (device_callback_) {
        auto device = getDevice(device_id);
        if (device) {
            device_callback_(*device);
        }
    }

    return device_removed;
}

uint32_t MultiChannelEngine::createBus(const std::string& name,
                                       const std::vector<uint32_t>& input_channels,
                                       const std::vector<uint32_t>& output_channels) {
    if (name.empty() || input_channels.empty() || output_channels.empty()) {
        return 0;
    }

    uint32_t bus_id = next_bus_id_++;

    auto bus_state = std::make_unique<BusState>();
    bus_state->config.id = bus_id;
    bus_state->config.name = name;
    bus_state->config.input_channels = input_channels;
    bus_state->config.output_channels = output_channels;
    bus_state->config.master_gain = 1.0f;
    bus_state->config.muted = false;
    bus_state->config.soloed = false;
    bus_state->config.routing_mode = AudioRoutingMode::DIRECT;

    // Initialize channel gains
    bus_state->config.channel_gains.resize(input_channels.size(), 1.0f);

    // Initialize buffers
    bus_state->mix_buffer.resize(session_.buffer_size * session_.channels, 0.0f);
    bus_state->output_buffer.resize(session_.buffer_size * session_.channels, 0.0f);

    std::lock_guard<std::mutex> lock(buses_mutex_);
    buses_[bus_id] = std::move(bus_state);

    return bus_id;
}

bool MultiChannelEngine::removeBus(uint32_t bus_id) {
    std::lock_guard<std::mutex> lock(buses_mutex_);

    auto it = buses_.find(bus_id);
    if (it == buses_.end()) {
        return false;
    }

    buses_.erase(it);
    return true;
}

std::optional<AudioBus> MultiChannelEngine::getBus(uint32_t bus_id) const {
    std::lock_guard<std::mutex> lock(buses_mutex_);

    auto it = buses_.find(bus_id);
    if (it == buses_.end()) {
        return std::nullopt;
    }

    return it->second->config;
}

bool MultiChannelEngine::addEffectsToBus(uint32_t bus_id, std::unique_ptr<RealtimeEffectsChain> effects) {
    if (!effects) {
        return false;
    }

    std::lock_guard<std::mutex> lock(buses_mutex_);

    auto it = buses_.find(bus_id);
    if (it == buses_.end()) {
        return false;
    }

    // Initialize effects chain if needed
    if (!effects->isInitialized()) {
        if (!effects->initialize(session_.sample_rate, session_.buffer_size, session_.channels)) {
            return false;
        }
    }

    it->second->config.effects_chain = std::move(effects);
    return true;
}

bool MultiChannelEngine::configureChannels(const std::vector<AudioChannel>& channels) {
    std::lock_guard<std::mutex> lock(channels_mutex_);

    channels_.clear();
    channels_.reserve(channels.size());

    for (const auto& channel : channels) {
        if (channel.index < 0 || channel.index >= session_.channels) {
            return false;
        }

        ChannelState state;
        state.config = channel;
        state.buffer.resize(session_.buffer_size, 0.0f);
        state.current_gain = channel.gain;
        state.target_gain = channel.gain;
        state.current_pan = channel.pan;
        state.target_pan = channel.pan;
        state.ramp_gain = false;
        state.ramp_pan = false;
        state.record_handle = 0;

        channels_.push_back(std::move(state));
    }

    // Ensure we have a state for each channel
    while (channels_.size() < static_cast<size_t>(session_.channels)) {
        ChannelState state;
        state.config.index = static_cast<int>(channels_.size());
        state.config.name = "Channel " + std::to_string(state.config.index);
        state.config.gain = 1.0f;
        state.config.pan = 0.0f;
        state.config.muted = false;
        state.config.soloed = false;
        state.buffer.resize(session_.buffer_size, 0.0f);
        state.current_gain = 1.0f;
        state.target_gain = 1.0f;
        state.current_pan = 0.0f;
        state.target_pan = 0.0f;
        state.ramp_gain = false;
        state.ramp_pan = false;
        state.record_handle = 0;

        channels_.push_back(std::move(state));
    }

    // Initialize routing matrix
    routing_matrix_.input_channels = session_.channels;
    routing_matrix_.output_channels = session_.channels;
    routing_matrix_.matrix.resize(session_.channels, std::vector<float>(session_.channels, 0.0f));

    // Set default routing (identity matrix)
    for (int i = 0; i < session_.channels; ++i) {
        routing_matrix_.matrix[i][i] = 1.0f;
    }

    return true;
}

std::optional<AudioChannel> MultiChannelEngine::getChannel(int channel_index) const {
    std::lock_guard<std::mutex> lock(channels_mutex_);

    if (channel_index < 0 || channel_index >= static_cast<int>(channels_.size())) {
        return std::nullopt;
    }

    return channels_[channel_index].config;
}

void MultiChannelEngine::setRoutingMatrix(const AudioRoutingMatrix& matrix) {
    std::lock_guard<std::mutex> lock(routing_mutex_);

    if (matrix.input_channels == session_.channels && matrix.output_channels == session_.channels) {
        routing_matrix_ = matrix;
    }
}

AudioRoutingMatrix MultiChannelEngine::getRoutingMatrix() const {
    std::lock_guard<std::mutex> lock(routing_mutex_);
    return routing_matrix_;
}

bool MultiChannelEngine::setChannelGain(int channel_index, float gain) {
    std::lock_guard<std::mutex> lock(channels_mutex_);

    if (channel_index < 0 || channel_index >= static_cast<int>(channels_.size())) {
        return false;
    }

    gain = std::max(MIN_GAIN, std::min(MAX_GAIN, gain));
    channels_[channel_index].config.gain = gain;
    channels_[channel_index].target_gain = gain;
    channels_[channel_index].ramp_gain = true;

    return true;
}

bool MultiChannelEngine::setChannelPan(int channel_index, float pan) {
    std::lock_guard<std::mutex> lock(channels_mutex_);

    if (channel_index < 0 || channel_index >= static_cast<int>(channels_.size())) {
        return false;
    }

    pan = std::max(-1.0f, std::min(1.0f, pan));
    channels_[channel_index].config.pan = pan;
    channels_[channel_index].target_pan = pan;
    channels_[channel_index].ramp_pan = true;

    return true;
}

bool MultiChannelEngine::setChannelMuted(int channel_index, bool muted) {
    std::lock_guard<std::mutex> lock(channels_mutex_);

    if (channel_index < 0 || channel_index >= static_cast<int>(channels_.size())) {
        return false;
    }

    channels_[channel_index].config.muted = muted;
    return true;
}

bool MultiChannelEngine::setChannelSoloed(int channel_index, bool soloed) {
    std::lock_guard<std::mutex> lock(channels_mutex_);

    if (channel_index < 0 || channel_index >= static_cast<int>(channels_.size())) {
        return false;
    }

    channels_[channel_index].config.soloed = soloed;
    return true;
}

bool MultiChannelEngine::setChannelRecording(int channel_index, bool enabled, const std::string& file_path) {
    std::lock_guard<std::mutex> lock(channels_mutex_);

    if (channel_index < 0 || channel_index >= static_cast<int>(channels_.size())) {
        return false;
    }

    channels_[channel_index].config.record_enabled = enabled;
    if (enabled) {
        channels_[channel_index].config.record_file_path = file_path;
    }

    // Recording implementation would go here
    return true;
}

bool MultiChannelEngine::setBusGain(uint32_t bus_id, float gain) {
    std::lock_guard<std::mutex> lock(buses_mutex_);

    auto it = buses_.find(bus_id);
    if (it == buses_.end()) {
        return false;
    }

    gain = std::max(MIN_GAIN, std::min(MAX_GAIN, gain));
    it->second->config.master_gain = gain;
    it->second->target_master_gain = gain;
    it->second->ramp_gain = true;

    return true;
}

bool MultiChannelEngine::setBusMuted(uint32_t bus_id, bool muted) {
    std::lock_guard<std::mutex> lock(buses_mutex_);

    auto it = buses_.find(bus_id);
    if (it == buses_.end()) {
        return false;
    }

    it->second->config.muted = muted;
    return true;
}

bool MultiChannelEngine::setBusSoloed(uint32_t bus_id, bool soloed) {
    std::lock_guard<std::mutex> lock(buses_mutex_);

    auto it = buses_.find(bus_id);
    if (it == buses_.end()) {
        return false;
    }

    it->second->config.soloed = soloed;
    return true;
}

bool MultiChannelEngine::setBusRecording(uint32_t bus_id, bool enabled, const std::string& file_path) {
    std::lock_guard<std::mutex> lock(buses_mutex_);

    auto it = buses_.find(bus_id);
    if (it == buses_.end()) {
        return false;
    }

    it->second->config.enable_recording = enabled;
    if (enabled) {
        it->second->config.record_file_path = file_path;
    }

    // Recording implementation would go here
    return true;
}

bool MultiChannelEngine::setChannelRouting(int from_channel, int to_channel, float gain) {
    std::lock_guard<std::mutex> lock(routing_mutex_);

    if (from_channel < 0 || from_channel >= routing_matrix_.input_channels ||
        to_channel < 0 || to_channel >= routing_matrix_.output_channels) {
        return false;
    }

    routing_matrix_.matrix[from_channel][to_channel] = gain;
    return true;
}

bool MultiChannelEngine::addSendRouting(int from_channel, uint32_t to_bus, float gain) {
    std::lock_guard<std::mutex> lock(routing_mutex_);

    if (from_channel < 0 || from_channel >= session_.channels) {
        return false;
    }

    // Ensure send_routing has space for this channel
    if (static_cast<size_t>(from_channel) >= send_routing_.size()) {
        send_routing_.resize(from_channel + 1);
    }

    // Check if this send already exists
    for (auto& [bus_id, send_gain] : send_routing_[from_channel]) {
        if (bus_id == to_bus) {
            send_gain = gain;
            return true;
        }
    }

    // Add new send
    send_routing_[from_channel].emplace_back(to_bus, gain);
    return true;
}

bool MultiChannelEngine::removeSendRouting(int from_channel, uint32_t to_bus) {
    std::lock_guard<std::mutex> lock(routing_mutex_);

    if (from_channel < 0 || from_channel >= static_cast<int>(send_routing_.size())) {
        return false;
    }

    auto& sends = send_routing_[from_channel];
    auto it = std::remove_if(sends.begin(), sends.end(),
                           [to_bus](const std::pair<uint32_t, float>& send) {
                               return send.first == to_bus;
                           });

    if (it != sends.end()) {
        sends.erase(it, sends.end());
        return true;
    }

    return false;
}

AudioEngineMetrics MultiChannelEngine::getMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void MultiChannelEngine::setDeviceCallback(AudioDeviceCallback callback) {
    device_callback_ = callback;
}

void MultiChannelEngine::setSessionCallback(AudioSessionCallback callback) {
    session_callback_ = callback;
}

void MultiChannelEngine::setMetricsCallback(AudioMetricsCallback callback) {
    metrics_callback_ = callback;
}

void MultiChannelEngine::setLevelCallback(AudioLevelCallback callback) {
    level_callback_ = callback;
}

void MultiChannelEngine::setChannelLayout(AudioChannelLayout layout) {
    std::lock_guard<std::mutex> lock(session_mutex_);

    session_.layout = layout;
    session_.channels = getChannelCountForLayout(layout);
    initializeChannelLayout();
}

AudioChannelLayout MultiChannelEngine::getChannelLayout() const {
    std::lock_guard<std::mutex> lock(session_mutex_);
    return session_.layout;
}

bool MultiChannelEngine::setBitDepth(AudioBitDepth bit_depth) {
    std::lock_guard<std::mutex> lock(session_mutex_);
    session_.bit_depth = bit_depth;
    return true;
}

AudioBitDepth MultiChannelEngine::getBitDepth() const {
    std::lock_guard<std::mutex> lock(session_mutex_);
    return session_.bit_depth;
}

void MultiChannelEngine::setDitheringEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(session_mutex_);
    session_.enable_dithering = enabled;
}

bool MultiChannelEngine::isDitheringEnabled() const {
    std::lock_guard<std::mutex> lock(session_mutex_);
    return session_.enable_dithering;
}

bool MultiChannelEngine::setSyncMode(AudioSyncMode mode) {
    std::lock_guard<std::mutex> lock(session_mutex_);
    session_.sync_mode = mode;
    return true;
}

AudioSyncMode MultiChannelEngine::getSyncMode() const {
    std::lock_guard<std::mutex> lock(session_mutex_);
    return session_.sync_mode;
}

bool MultiChannelEngine::validateConfiguration() const {
    return session_.sample_rate > 0 &&
           session_.buffer_size > 0 &&
           session_.channels > 0 &&
           session_.channels <= 64; // Reasonable maximum
}

void MultiChannelEngine::reset() {
    if (!initialized_) {
        return;
    }

    // Stop processing
    bool was_running = running_;
    if (was_running) {
        stop();
    }

    // Reset all channels
    {
        std::lock_guard<std::mutex> lock(channels_mutex_);
        for (auto& channel : channels_) {
            channel.current_gain = channel.config.gain;
            channel.target_gain = channel.config.gain;
            channel.current_pan = channel.config.pan;
            channel.target_pan = channel.config.pan;
            channel.ramp_gain = false;
            channel.ramp_pan = false;
            std::fill(channel.buffer.begin(), channel.buffer.end(), 0.0f);
        }
    }

    // Reset all buses
    {
        std::lock_guard<std::mutex> lock(buses_mutex_);
        for (auto& [id, bus] : buses_) {
            bus->current_master_gain = bus->config.master_gain;
            bus->target_master_gain = bus->config.master_gain;
            bus->ramp_gain = false;
            std::fill(bus->mix_buffer.begin(), bus->mix_buffer.end(), 0.0f);
            std::fill(bus->output_buffer.begin(), bus->output_buffer.end(), 0.0f);
        }
    }

    // Reset metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        std::memset(&metrics_, 0, sizeof(metrics_));
        xruns_count_ = 0;
    }

    // Clear buffers
    std::fill(input_buffer_.begin(), input_buffer_.end(), 0.0f);
    std::fill(output_buffer_.begin(), output_buffer_.end(), 0.0f);
    std::fill(mix_buffer_.begin(), mix_buffer_.end(), 0.0f);
    std::fill(temp_buffer_.begin(), temp_buffer_.end(), 0.0f);
    std::fill(peak_levels_.begin(), peak_levels_.end(), 0.0f);

    // Reset processing state
    processed_frames_ = 0;

    // Restart if it was running
    if (was_running) {
        start();
    }
}

bool MultiChannelEngine::initializeDevices() {
    // Scan for available devices
    scanDevices();

    // Validate device compatibility
    std::vector<AudioDevice> test_devices;

    if (master_device_id_ != 0) {
        auto master = getDevice(master_device_id_);
        if (master) {
            test_devices.push_back(*master);
        }
    }

    for (uint32_t device_id : input_device_ids_) {
        auto device = getDevice(device_id);
        if (device) {
            test_devices.push_back(*device);
        }
    }

    for (uint32_t device_id : output_device_ids_) {
        auto device = getDevice(device_id);
        if (device) {
            test_devices.push_back(*device);
        }
    }

    return validateDeviceCompatibility(test_devices);
}

bool MultiChannelEngine::initializeAudioDrivers() {
    // Platform-specific audio driver initialization would go here
    // This is a simplified implementation

#ifdef _WIN32
    // Initialize WASAPI or ASIO
    return true;
#elif __APPLE__
    // Initialize CoreAudio
    return true;
#elif __linux__
    // Initialize ALSA or JACK
    return true;
#else
    return true;
#endif
}

void MultiChannelEngine::shutdownAudioDrivers() {
    // Platform-specific audio driver shutdown
}

bool MultiChannelEngine::validateDeviceCompatibility(const std::vector<AudioDevice>& devices) const {
    if (devices.empty()) {
        return false;
    }

    // Check if all devices support the session sample rate
    for (const auto& device : devices) {
        if (std::find(device.supported_sample_rates.begin(), device.supported_sample_rates.end(),
                     session_.sample_rate) == device.supported_sample_rates.end()) {
            return false;
        }

        if (std::find(device.supported_bit_depths.begin(), device.supported_bit_depths.end(),
                     session_.bit_depth) == device.supported_bit_depths.end()) {
            return false;
        }
    }

    return true;
}

void MultiChannelEngine::processChannels(float* input_buffer, float* output_buffer, size_t num_samples) {
    std::lock_guard<std::mutex> lock(channels_mutex_);

    // Check for soloed channels
    bool any_soloed = false;
    for (const auto& channel : channels_) {
        if (channel.config.soloed) {
            any_soloed = true;
            break;
        }
    }

    // Process each channel
    for (size_t ch = 0; ch < channels_.size() && ch < static_cast<size_t>(session_.channels); ++ch) {
        auto& channel = channels_[ch];

        // Check if channel should be processed
        bool should_process = !channel.config.muted;
        if (any_soloed) {
            should_process = channel.config.soloed;
        }

        if (!should_process) {
            std::fill(channel.buffer.begin(), channel.buffer.end(), 0.0f);
            continue;
        }

        // Extract channel data
        for (size_t i = 0; i < num_samples; ++i) {
            channel.buffer[i] = input_buffer[i * session_.channels + ch];
        }

        // Apply gain ramping if needed
        if (channel.ramp_gain) {
            float gain_diff = channel.target_gain - channel.current_gain;
            float gain_step = gain_diff / static_cast<float>(num_samples);

            for (size_t i = 0; i < num_samples; ++i) {
                channel.current_gain += gain_step;
                channel.buffer[i] *= channel.current_gain;
            }

            if (std::abs(channel.current_gain - channel.target_gain) < 0.001f) {
                channel.current_gain = channel.target_gain;
                channel.ramp_gain = false;
            }
        } else {
            // Apply current gain
            for (size_t i = 0; i < num_samples; ++i) {
                channel.buffer[i] *= channel.current_gain;
            }
        }

        // Apply pan for stereo
        if (session_.channels >= 2) {
            if (channel.ramp_pan) {
                float pan_diff = channel.target_pan - channel.current_pan;
                float pan_step = pan_diff / static_cast<float>(num_samples);

                for (size_t i = 0; i < num_samples; ++i) {
                    channel.current_pan += pan_step;
                    float left_gain = std::cos((channel.current_pan + 1.0f) * M_PI / 4.0f);
                    float right_gain = std::sin((channel.current_pan + 1.0f) * M_PI / 4.0f);

                    output_buffer[i * 2] += channel.buffer[i] * left_gain;
                    output_buffer[i * 2 + 1] += channel.buffer[i] * right_gain;
                }

                if (std::abs(channel.current_pan - channel.target_pan) < 0.001f) {
                    channel.current_pan = channel.target_pan;
                    channel.ramp_pan = false;
                }
            } else {
                float left_gain = std::cos((channel.current_pan + 1.0f) * M_PI / 4.0f);
                float right_gain = std::sin((channel.current_pan + 1.0f) * M_PI / 4.0f);

                for (size_t i = 0; i < num_samples; ++i) {
                    output_buffer[i * 2] += channel.buffer[i] * left_gain;
                    output_buffer[i * 2 + 1] += channel.buffer[i] * right_gain;
                }
            }
        } else {
            // Mono output
            for (size_t i = 0; i < num_samples; ++i) {
                output_buffer[i] += channel.buffer[i];
            }
        }
    }
}

void MultiChannelEngine::processBuses(float* output_buffer, size_t num_samples) {
    std::lock_guard<std::mutex> lock(buses_mutex_);

    // Check for soloed buses
    bool any_soloed = false;
    for (const auto& [id, bus] : buses_) {
        if (bus->config.soloed) {
            any_soloed = true;
            break;
        }
    }

    // Process each bus
    for (const auto& [bus_id, bus] : buses_) {
        // Check if bus should be processed
        bool should_process = !bus->config.muted;
        if (any_soloed) {
            should_process = bus->config.soloed;
        }

        if (!should_process) {
            std::fill(bus->mix_buffer.begin(), bus->mix_buffer.end(), 0.0f);
            continue;
        }

        // Mix input channels into bus
        std::fill(bus->mix_buffer.begin(), bus->mix_buffer.end(), 0.0f);

        for (size_t ch_idx = 0; ch_idx < bus->config.input_channels.size(); ++ch_idx) {
            uint32_t input_ch = bus->config.input_channels[ch_idx];
            float channel_gain = bus->config.channel_gains[ch_idx];

            if (input_ch < static_cast<uint32_t>(channels_.size())) {
                const auto& channel = channels_[input_ch];

                for (size_t i = 0; i < num_samples; ++i) {
                    for (size_t out_ch = 0; out_ch < static_cast<size_t>(session_.channels); ++out_ch) {
                        bus->mix_buffer[i * session_.channels + out_ch] +=
                            channel.buffer[i] * channel_gain;
                    }
                }
            }
        }

        // Apply master gain ramping
        if (bus->ramp_gain) {
            float gain_diff = bus->target_master_gain - bus->current_master_gain;
            float gain_step = gain_diff / static_cast<float>(num_samples);

            for (size_t i = 0; i < num_samples; ++i) {
                bus->current_master_gain += gain_step;
                for (size_t ch = 0; ch < static_cast<size_t>(session_.channels); ++ch) {
                    bus->mix_buffer[i * session_.channels + ch] *= bus->current_master_gain;
                }
            }

            if (std::abs(bus->current_master_gain - bus->target_master_gain) < 0.001f) {
                bus->current_master_gain = bus->target_master_gain;
                bus->ramp_gain = false;
            }
        } else {
            // Apply current master gain
            for (size_t i = 0; i < num_samples * session_.channels; ++i) {
                bus->mix_buffer[i] *= bus->current_master_gain;
            }
        }

        // Apply effects if available
        if (bus->config.effects_chain) {
            bus->config.effects_chain->processAudio(bus->mix_buffer.data(), num_samples);
        }

        // Route to output channels
        for (size_t out_ch_idx = 0; out_ch_idx < bus->config.output_channels.size(); ++out_ch_idx) {
            uint32_t output_ch = bus->config.output_channels[out_ch_idx];

            if (output_ch < static_cast<uint32_t>(session_.channels)) {
                for (size_t i = 0; i < num_samples; ++i) {
                    output_buffer[i * session_.channels + output_ch] +=
                        bus->mix_buffer[i * session_.channels + output_ch];
                }
            }
        }
    }
}

void MultiChannelEngine::applyRouting(float* input_buffer, float* output_buffer, size_t num_samples) {
    std::lock_guard<std::mutex> lock(routing_mutex_);

    if (!routing_matrix_.enabled) {
        return;
    }

    // Clear temp buffer
    std::fill(temp_buffer_.begin(), temp_buffer_.end(), 0.0f);

    // Apply routing matrix
    for (int in_ch = 0; in_ch < routing_matrix_.input_channels; ++in_ch) {
        for (int out_ch = 0; out_ch < routing_matrix_.output_channels; ++out_ch) {
            float gain = routing_matrix_.matrix[in_ch][out_ch];
            if (std::abs(gain) > 0.0001f) {
                for (size_t i = 0; i < num_samples; ++i) {
                    temp_buffer_[i * session_.channels + out_ch] +=
                        input_buffer[i * session_.channels + in_ch] * gain;
                }
            }
        }
    }

    // Add routed signal to output
    for (size_t i = 0; i < num_samples * session_.channels; ++i) {
        output_buffer[i] += temp_buffer_[i];
    }

    // Apply send routing
    for (size_t ch = 0; ch < send_routing_.size() && ch < static_cast<size_t>(session_.channels); ++ch) {
        for (const auto& [bus_id, gain] : send_routing_[ch]) {
            auto bus_it = buses_.find(bus_id);
            if (bus_it != buses_.end() && std::abs(gain) > 0.0001f) {
                for (size_t i = 0; i < num_samples; ++i) {
                    bus_it->second->mix_buffer[i * session_.channels] +=
                        input_buffer[i * session_.channels + ch] * gain;
                }
            }
        }
    }
}

void MultiChannelEngine::applyDithering(float* buffer, size_t num_samples) {
    if (!session_.enable_dithering) {
        return;
    }

    // Simple triangular PDF dithering
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

    for (size_t i = 0; i < num_samples * session_.channels; ++i) {
        float dither = (dis(gen) + dis(gen)) / 2.0f; // Triangular PDF
        buffer[i] += dither * (1.0f / 65536.0f); // Scale for 16-bit target
    }
}

void MultiChannelEngine::updateChannelLevels(float* buffer, size_t num_samples) {
    std::vector<float> current_levels(session_.channels, 0.0f);

    // Calculate RMS levels for each channel
    for (int ch = 0; ch < session_.channels; ++ch) {
        float sum = 0.0f;
        for (size_t i = 0; i < num_samples; ++i) {
            float sample = buffer[i * session_.channels + ch];
            sum += sample * sample;
        }

        current_levels[ch] = std::sqrt(sum / static_cast<float>(num_samples));
        peak_levels_[ch] = std::max(peak_levels_[ch], current_levels[ch]);

        // Decay peak levels
        peak_levels_[ch] *= 0.999f;
    }

    // Update metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.input_levels = current_levels;
        metrics_.output_levels = current_levels;
    }

    // Notify level callback
    if (level_callback_) {
        for (int ch = 0; ch < session_.channels; ++ch) {
            level_callback_(ch, current_levels[ch]);
        }
    }
}

void MultiChannelEngine::updateMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    // Update counts
    metrics_.active_channels = 0;
    {
        std::lock_guard<std::mutex> ch_lock(channels_mutex_);
        for (const auto& channel : channels_) {
            if (!channel.config.muted) {
                metrics_.active_channels++;
            }
        }
    }

    metrics_.active_buses = 0;
    {
        std::lock_guard<std::mutex> bus_lock(buses_mutex_);
        for (const auto& [id, bus] : buses_) {
            if (!bus->config.muted) {
                metrics_.active_buses++;
            }
        }
    }

    // Update timing metrics
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(now - start_time_).count();
    if (elapsed > 0.0) {
        metrics_.processed_frames = processed_frames_;
        double expected_frames = elapsed * session_.sample_rate / session_.buffer_size;
        metrics_.dropped_frames = static_cast<uint64_t>(expected_frames) - processed_frames_;
        metrics_.real_time_stable = metrics_.dropped_frames == 0;
    }

    // Update memory usage (simplified)
    metrics_.memory_usage_mb = (input_buffer_.size() + output_buffer_.size() +
                                mix_buffer_.size() + temp_buffer_.size()) * sizeof(float) / (1024.0 * 1024.0);

    metrics_.last_update = std::chrono::steady_clock::now();

    // Notify callback
    if (metrics_callback_) {
        metrics_callback_(metrics_);
    }
}

void MultiChannelEngine::startRecording() {
    // Recording implementation would go here
}

void MultiChannelEngine::stopRecording() {
    // Recording implementation would go here
}

void MultiChannelEngine::initializeChannelLayout() {
    // Initialize default channel configuration based on layout
    std::vector<AudioChannel> channels;
    int channel_count = getChannelCountForLayout(session_.layout);

    for (int i = 0; i < channel_count; ++i) {
        AudioChannel channel;
        channel.index = i;
        channel.name = "Channel " + std::to_string(i);
        channel.gain = 1.0f;
        channel.pan = 0.0f;
        channel.muted = false;
        channel.soloed = false;

        channels.push_back(channel);
    }

    configureChannels(channels);
}

int MultiChannelEngine::getChannelCountForLayout(AudioChannelLayout layout) const {
    switch (layout) {
        case AudioChannelLayout::MONO: return 1;
        case AudioChannelLayout::STEREO: return 2;
        case AudioChannelLayout::TWO_POINT_ONE: return 3;
        case AudioChannelLayout::THREE_POINT_ONE: return 4;
        case AudioChannelLayout::FIVE_POINT_ONE: return 6;
        case AudioChannelLayout::SEVEN_POINT_ONE: return 8;
        case AudioChannelLayout::SEVEN_POINT_ONE_FOUR: return 12;
        case AudioChannelLayout::NINE_POINT_ONE_FOUR: return 14;
        case AudioChannelLayout::DOLBY_ATMOS: return 16; // Approximate
        case AudioChannelLayout::DTS_X: return 16; // Approximate
        case AudioChannelLayout::AURO_3D: return 14; // Approximate
        case AudioChannelLayout::CUSTOM: return session_.channels; // Use current setting
        default: return 2;
    }
}

std::vector<float> MultiChannelEngine::getChannelGainsForLayout(AudioChannelLayout layout) const {
    // Return default channel gains for various layouts
    switch (layout) {
        case AudioChannelLayout::MONO:
            return {1.0f};

        case AudioChannelLayout::STEREO:
            return {1.0f, 1.0f};

        case AudioChannelLayout::FIVE_POINT_ONE:
            return {1.0f, 1.0f, 1.0f, 0.707f, 0.707f, 0.707f};

        case AudioChannelLayout::SEVEN_POINT_ONE:
            return {1.0f, 1.0f, 1.0f, 0.707f, 0.707f, 0.707f, 0.707f, 0.707f};

        default:
            return std::vector<float>(getChannelCountForLayout(layout), 1.0f);
    }
}

void MultiChannelEngine::performDeviceCrossfade(uint32_t old_device_id, uint32_t new_device_id) {
    if (!session_.auto_crossfade_devices) {
        return;
    }

    float fade_duration = session_.crossfade_duration_ms / 1000.0f;
    int fade_samples = static_cast<int>(fade_duration * session_.sample_rate);

    // Crossfade implementation would go here
    // This would involve gradually reducing volume on old device and increasing on new device
}

bool MultiChannelEngine::startDevice(uint32_t device_id) {
    auto device = getDevice(device_id);
    if (!device) {
        return false;
    }

    // Platform-specific device start implementation would go here
    return true;
}

bool MultiChannelEngine::stopDevice(uint32_t device_id) {
    auto device = getDevice(device_id);
    if (!device) {
        return false;
    }

    // Platform-specific device stop implementation would go here
    return true;
}

void MultiChannelEngine::processDeviceAudio(AudioDevice& device, float* buffer, size_t num_samples, bool is_input) {
    // Device-specific audio processing would go here
    // This handles actual I/O with the audio hardware
}

float MultiChannelEngine::calculateRMS(float* buffer, size_t samples) {
    float sum = 0.0f;
    for (size_t i = 0; i < samples; ++i) {
        sum += buffer[i] * buffer[i];
    }
    return std::sqrt(sum / static_cast<float>(samples));
}

float MultiChannelEngine::calculatePeak(float* buffer, size_t samples) {
    float peak = 0.0f;
    for (size_t i = 0; i < samples; ++i) {
        peak = std::max(peak, std::abs(buffer[i]));
    }
    return peak;
}

void MultiChannelEngine::applyGainRamp(float* buffer, size_t samples, float start_gain, float end_gain) {
    float gain_diff = end_gain - start_gain;
    float gain_step = gain_diff / static_cast<float>(samples);

    for (size_t i = 0; i < samples; ++i) {
        float current_gain = start_gain + gain_step * i;
        buffer[i] *= current_gain;
    }
}

void MultiChannelEngine::applyPan(float* buffer, size_t samples, float pan, int channels) {
    if (channels < 2) {
        return;
    }

    float left_gain = std::cos((pan + 1.0f) * M_PI / 4.0f);
    float right_gain = std::sin((pan + 1.0f) * M_PI / 4.0f);

    for (size_t i = 0; i < samples; ++i) {
        float sample = buffer[i];
        buffer[i * 2] = sample * left_gain;
        buffer[i * 2 + 1] = sample * right_gain;
    }
}

void MultiChannelEngine::syncAllDevices() {
    // Device synchronization implementation would go here
}

void MultiChannelEngine::detectXRuns() {
    // XRUN detection implementation would go here
    // This would monitor buffer overflows/underflows
}

// AudioDeviceManager implementation
AudioDeviceManager::AudioDeviceManager() : initialized_(false) {}

AudioDeviceManager::~AudioDeviceManager() {
    unloadDeviceDrivers();
}

bool AudioDeviceManager::initialize() {
    if (initialized_) {
        return true;
    }

    if (!loadDeviceDrivers()) {
        return false;
    }

    initialized_ = true;
    return true;
}

std::vector<AudioDevice> AudioDeviceManager::scanDevices() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return {};
    }

    // Platform-specific device scanning
    std::vector<AudioDevice> devices;

#ifdef _WIN32
    // Windows device enumeration
    scanWindowsDevices(devices);
#elif __APPLE__
    // macOS device enumeration
    scanMacOSDevices(devices);
#elif __linux__
    // Linux device enumeration
    scanLinuxDevices(devices);
#endif

    // Update internal device list
    devices_.clear();
    for (auto& device : devices) {
        devices_[device.id] = device;
    }

    return devices;
}

std::vector<int> AudioDeviceManager::getDeviceSupportedSampleRates(uint32_t device_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = devices_.find(device_id);
    if (it != devices_.end()) {
        return it->second.supported_sample_rates;
    }

    return {};
}

std::vector<AudioBitDepth> AudioDeviceManager::getDeviceSupportedBitDepths(uint32_t device_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = devices_.find(device_id);
    if (it != devices_.end()) {
        return it->second.supported_bit_depths;
    }

    return {};
}

bool AudioDeviceManager::testDevice(uint32_t device_id, int sample_rate, int buffer_size) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = devices_.find(device_id);
    if (it == devices_.end()) {
        return false;
    }

    // Device testing implementation would go here
    return true;
}

uint32_t AudioDeviceManager::getDefaultInputDevice() const {
    // Return default input device ID
    // Platform-specific implementation would go here
    return 0;
}

uint32_t AudioDeviceManager::getDefaultOutputDevice() const {
    // Return default output device ID
    // Platform-specific implementation would go here
    return 0;
}

bool AudioDeviceManager::loadDeviceDrivers() {
    // Load platform-specific audio drivers
#ifdef _WIN32
    return initializeWASAPI();
#elif __APPLE__
    return initializeCoreAudio();
#elif __linux__
    return initializeALSA();
#endif
    return true;
}

void AudioDeviceManager::unloadDeviceDrivers() {
    // Unload platform-specific audio drivers
}

bool AudioDeviceManager::initializeASIO() {
    // ASIO initialization for Windows
    return true;
}

bool AudioDeviceManager::initializeCoreAudio() {
    // CoreAudio initialization for macOS
    return true;
}

bool AudioDeviceManager::initializeWASAPI() {
    // WASAPI initialization for Windows
    return true;
}

bool AudioDeviceManager::initializeALSA() {
    // ALSA initialization for Linux
    return true;
}

bool AudioDeviceManager::initializeJack() {
    // JACK initialization for Linux
    return true;
}

// Utility functions implementation
namespace audio_engine_utils {

int getChannelCount(AudioChannelLayout layout) {
    switch (layout) {
        case AudioChannelLayout::MONO: return 1;
        case AudioChannelLayout::STEREO: return 2;
        case AudioChannelLayout::TWO_POINT_ONE: return 3;
        case AudioChannelLayout::THREE_POINT_ONE: return 4;
        case AudioChannelLayout::FIVE_POINT_ONE: return 6;
        case AudioChannelLayout::SEVEN_POINT_ONE: return 8;
        case AudioChannelLayout::SEVEN_POINT_ONE_FOUR: return 12;
        case AudioChannelLayout::NINE_POINT_ONE_FOUR: return 14;
        case AudioChannelLayout::DOLBY_ATMOS: return 16;
        case AudioChannelLayout::DTS_X: return 16;
        case AudioChannelLayout::AURO_3D: return 14;
        case AudioChannelLayout::CUSTOM: return 2; // Default fallback
        default: return 2;
    }
}

std::vector<std::string> getChannelNames(AudioChannelLayout layout) {
    switch (layout) {
        case AudioChannelLayout::MONO:
            return {"Center"};

        case AudioChannelLayout::STEREO:
            return {"Left", "Right"};

        case AudioChannelLayout::TWO_POINT_ONE:
            return {"Left", "Right", "LFE"};

        case AudioChannelLayout::THREE_POINT_ONE:
            return {"Left", "Right", "Center", "LFE"};

        case AudioChannelLayout::FIVE_POINT_ONE:
            return {"Left", "Right", "Center", "LFE", "Left Surround", "Right Surround"};

        case AudioChannelLayout::SEVEN_POINT_ONE:
            return {"Left", "Right", "Center", "LFE", "Left Surround", "Right Surround",
                    "Left Rear", "Right Rear"};

        default:
            return {};
    }
}

std::vector<float> getDefaultChannelPanning(AudioChannelLayout layout) {
    int count = getChannelCount(layout);
    std::vector<float> panning(count, 0.0f);

    // Set default panning based on layout
    switch (layout) {
        case AudioChannelLayout::STEREO:
            panning[0] = -1.0f; // Left
            panning[1] = 1.0f;  // Right
            break;
        default:
            break;
    }

    return panning;
}

std::vector<float> getDefaultSpeakerPositions(AudioChannelLayout layout) {
    // Return default speaker positions in 3D space
    switch (layout) {
        case AudioChannelLayout::STEREO:
            return {-1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}; // Left, Right
        case AudioChannelLayout::FIVE_POINT_ONE:
            return {-1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // L, R
                    0.0f, 0.0f, 1.0f, 0.0f, 0.0f, -1.0f, // C, LFE
                    -0.7f, 0.0f, -0.7f, 0.7f, 0.0f, -0.7f}; // LS, RS
        default:
            return {};
    }
}

int getBitDepthSize(AudioBitDepth bit_depth) {
    switch (bit_depth) {
        case AudioBitDepth::INT16: return 2;
        case AudioBitDepth::INT24: return 3;
        case AudioBitDepth::INT32: return 4;
        case AudioBitDepth::FLOAT32: return 4;
        case AudioBitDepth::FLOAT64: return 8;
        case AudioBitDepth::DSD64:
        case AudioBitDepth::DSD128:
        case AudioBitDepth::DSD256:
        case AudioBitDepth::DSD512:
            return 1; // 1-bit DSD
        default: return 2;
    }
}

double getBitDepthDynamicRange(AudioBitDepth bit_depth) {
    switch (bit_depth) {
        case AudioBitDepth::INT16: return 96.0; // 16-bit theoretical
        case AudioBitDepth::INT24: return 144.0; // 24-bit theoretical
        case AudioBitDepth::INT32: return 192.0; // 32-bit theoretical
        case AudioBitDepth::FLOAT32: return 144.0; // 24-bit mantissa
        case AudioBitDepth::FLOAT64: return 192.0; // 53-bit mantissa
        default: return 96.0;
    }
}

bool requiresDithering(AudioBitDepth source, AudioBitDepth target) {
    int source_bits = getBitDepthSize(source) * 8;
    int target_bits = getBitDepthSize(target) * 8;

    // Dither when going from higher bit depth to lower integer bit depth
    return source_bits > target_bits && target == AudioBitDepth::INT16;
}

std::string getDeviceTypeString(AudioDeviceType type) {
    switch (type) {
        case AudioDeviceType::INPUT: return "Input";
        case AudioDeviceType::OUTPUT: return "Output";
        case AudioDeviceType::INPUT_OUTPUT: return "Input/Output";
        case AudioDeviceType::VIRTUAL: return "Virtual";
        case AudioDeviceType::NETWORK: return "Network";
        default: return "Unknown";
    }
}

std::string getLayoutString(AudioChannelLayout layout) {
    switch (layout) {
        case AudioChannelLayout::MONO: return "Mono";
        case AudioChannelLayout::STEREO: return "Stereo";
        case AudioChannelLayout::TWO_POINT_ONE: return "2.1";
        case AudioChannelLayout::THREE_POINT_ONE: return "3.1";
        case AudioChannelLayout::FIVE_POINT_ONE: return "5.1";
        case AudioChannelLayout::SEVEN_POINT_ONE: return "7.1";
        case AudioChannelLayout::SEVEN_POINT_ONE_FOUR: return "7.1.4";
        case AudioChannelLayout::NINE_POINT_ONE_FOUR: return "9.1.4";
        case AudioChannelLayout::DOLBY_ATMOS: return "Dolby Atmos";
        case AudioChannelLayout::DTS_X: return "DTS:X";
        case AudioChannelLayout::AURO_3D: return "Auro-3D";
        case AudioChannelLayout::CUSTOM: return "Custom";
        default: return "Unknown";
    }
}

std::string getBitDepthString(AudioBitDepth bit_depth) {
    switch (bit_depth) {
        case AudioBitDepth::INT16: return "16-bit";
        case AudioBitDepth::INT24: return "24-bit";
        case AudioBitDepth::INT32: return "32-bit";
        case AudioBitDepth::FLOAT32: return "32-bit Float";
        case AudioBitDepth::FLOAT64: return "64-bit Float";
        case AudioBitDepth::DSD64: return "DSD64";
        case AudioBitDepth::DSD128: return "DSD128";
        case AudioBitDepth::DSD256: return "DSD256";
        case AudioBitDepth::DSD512: return "DSD512";
        default: return "Unknown";
    }
}

std::string getSyncModeString(AudioSyncMode mode) {
    switch (mode) {
        case AudioSyncMode::NONE: return "None";
        case AudioSyncMode::CLOCK_MASTER: return "Clock Master";
        case AudioSyncMode::CLOCK_SLAVE: return "Clock Slave";
        case AudioSyncMode::WORD_CLOCK: return "Word Clock";
        case AudioSyncMode::ADAT_SYNC: return "ADAT Sync";
        case AudioSyncMode::SPDIF_SYNC: return "S/PDIF Sync";
        case AudioSyncMode::MTC: return "MIDI Time Code";
        case AudioSyncMode::LTC: return "Linear Time Code";
        case AudioSyncMode::JAM_SYNC: return "Jam Sync";
        default: return "Unknown";
    }
}

void convertBitDepth(const void* input, void* output, size_t samples,
                    AudioBitDepth from, AudioBitDepth to, bool dither) {
    // Bit depth conversion implementation would go here
    // This is a simplified placeholder

    if (from == to) {
        std::memcpy(output, input, samples * getBitDepthSize(from));
        return;
    }

    // Convert through float32 for simplicity
    std::vector<float> float_samples(samples);

    // Convert input to float32
    switch (from) {
        case AudioBitDepth::INT16: {
            const int16_t* src = static_cast<const int16_t*>(input);
            for (size_t i = 0; i < samples; ++i) {
                float_samples[i] = static_cast<float>(src[i]) / 32768.0f;
            }
            break;
        }
        case AudioBitDepth::FLOAT32: {
            const float* src = static_cast<const float*>(input);
            std::copy(src, src + samples, float_samples.begin());
            break;
        }
        default:
            // Unsupported conversion
            break;
    }

    // Convert float32 to output
    switch (to) {
        case AudioBitDepth::INT16: {
            int16_t* dst = static_cast<int16_t*>(output);
            for (size_t i = 0; i < samples; ++i) {
                dst[i] = static_cast<int16_t>(float_samples[i] * 32767.0f);
            }
            break;
        }
        case AudioBitDepth::FLOAT32: {
            float* dst = static_cast<float*>(output);
            std::copy(float_samples.begin(), float_samples.end(), dst);
            break;
        }
        default:
            // Unsupported conversion
            break;
    }
}

void interleaveChannels(const float** input, float* output, size_t samples, int channels) {
    for (size_t i = 0; i < samples; ++i) {
        for (int ch = 0; ch < channels; ++ch) {
            output[i * channels + ch] = input[ch][i];
        }
    }
}

void deinterleaveChannels(const float* input, float** output, size_t samples, int channels) {
    for (size_t i = 0; i < samples; ++i) {
        for (int ch = 0; ch < channels; ++ch) {
            output[ch][i] = input[i * channels + ch];
        }
    }
}

float calculateLUFS(const float* buffer, size_t samples, int sample_rate) {
    // LUFS calculation implementation would go here
    // This is a simplified placeholder

    float sum = 0.0f;
    for (size_t i = 0; i < samples; ++i) {
        sum += buffer[i] * buffer[i];
    }

    float rms = std::sqrt(sum / static_cast<float>(samples));
    return 20.0f * std::log10(rms + 1e-10f) - 0.691f; // LUFS approximation
}

float calculateTruePeak(const float* buffer, size_t samples, int sample_rate) {
    // True peak calculation with oversampling would go here
    // This is a simplified placeholder

    float peak = 0.0f;
    for (size_t i = 0; i < samples; ++i) {
        peak = std::max(peak, std::abs(buffer[i]));
    }

    return 20.0f * std::log10(peak + 1e-10f);
}

std::vector<float> calculateFrequencyBands(const float* buffer, size_t samples, int sample_rate,
                                          const std::vector<float>& frequencies) {
    // Frequency band analysis would go here
    // This would typically use FFT or band-pass filters
    std::vector<float> levels(frequencies.size(), 0.0f);

    // Simplified placeholder implementation
    for (size_t i = 0; i < frequencies.size(); ++i) {
        levels[i] = -60.0f; // Default noise floor
    }

    return levels;
}

} // namespace audio_engine_utils

} // namespace audio
} // namespace core
} // namespace vortex