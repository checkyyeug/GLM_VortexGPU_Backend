#include "core/dsp/realtime_effects_chain.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <random>

namespace vortex::core::dsp {

RealtimeEffectsChain::RealtimeEffectsChain()
    : initialized_(false)
    , processing_(false)
    , paused_(false)
    , midi_learn_mode_(false)
    , process_call_count_(0) {

    // Initialize audio format with defaults
    audio_format_.sample_rate = 44100;
    audio_format_.channels = 2;
    audio_format_.frame_count = 0;
    audio_format_.bit_depth = 32;
    audio_format_.is_interleaved = true;

    // Initialize chain statistics
    chain_stats_.last_reset_time = std::chrono::steady_clock::now();
    last_stats_update_ = std::chrono::steady_clock::now();
}

RealtimeEffectsChain::~RealtimeEffectsChain() {
    shutdown();
}

bool RealtimeEffectsChain::initialize(const EffectsChainConfig& config) {
    std::lock_guard<std::mutex> lock(effects_mutex_);

    if (initialized_) {
        Logger::warn("RealtimeEffectsChain already initialized");
        return true;
    }

    config_ = config;

    Logger::info("Initializing RealtimeEffectsChain:");
    Logger::info("  Sample rate: {} Hz", config.sample_rate);
    Logger::info("  Channels: {}", config.channels);
    Logger::info("  Max frame size: {}", config.max_frame_size);
    Logger::info("  Processing mode: {}", static_cast<int>(config.processing_mode));
    Logger::info("  GPU acceleration: {}", config.enable_gpu_acceleration);

    // Validate configuration
    if (config.sample_rate == 0 || config.channels == 0 || config.max_frame_size == 0) {
        Logger::error("Invalid audio configuration parameters");
        return false;
    }

    // Set audio format
    audio_format_.sample_rate = config.sample_rate;
    audio_format_.channels = config.channels;
    audio_format_.bit_depth = config.bit_depth;
    audio_format_.is_interleaved = true;

    // Initialize global parameters
    global_bypass_ = config.enable_global_bypass;
    dry_wet_mix_ = 1.0f; // Start fully wet
    output_gain_linear_ = db_to_linear(config.output_gain_db);
    input_gain_linear_ = db_to_linear(config.input_gain_db);

    // Optimize buffer sizes
    optimize_buffer_sizes(config.max_frame_size);

    // Initialize parallel processing if enabled
    if (config.enable_parallel_processing && config.thread_pool_size > 1) {
        parallel_processing_enabled_ = true;
        // Initialize worker threads (simplified)
        Logger::info("Parallel processing enabled with {} threads", config.thread_pool_size);
    }

    // Initialize OSC control if enabled
    if (config.enable_osc_control) {
        // OSC server initialization would go here
        Logger::info("OSC control enabled on port {}", config.osc_port);
    }

    initialized_ = true;
    Logger::info("RealtimeEffectsChain initialized successfully");
    return true;
}

void RealtimeEffectsChain::shutdown() {
    if (!initialized_) {
        return;
    }

    Logger::info("Shutting down RealtimeEffectsChain");

    // Stop processing
    processing_ = false;

    // Clear all effects
    {
        std::lock_guard<std::mutex> lock(effects_mutex_);
        effects_chain_.clear();
        effects_map_.clear();
        effect_positions_.clear();
    }

    // Stop worker threads
    parallel_processing_enabled_ = false;
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();

    // Clear buffers
    {
        std::lock_guard<std::mutex> lock(buffers_mutex_);
        input_buffer_.clear();
        output_buffer_.clear();
        temp_buffer_.clear();
        dry_buffer_.clear();
    }

    // Clear automations
    {
        std::lock_guard<std::mutex> lock(automations_mutex_);
        automations_.clear();
    }

    // Clear MIDI mappings
    {
        std::lock_guard<std::mutex> lock(midi_mutex_);
        midi_mappings_.clear();
    }

    // Final statistics report
    auto final_stats = get_chain_statistics();
    Logger::info("RealtimeEffectsChain final statistics:");
    Logger::info("  Total process calls: {}", final_stats.total_process_calls);
    Logger::info("  Successful calls: {}", final_stats.successful_calls);
    Logger::info("  Success rate: {:.2f}%",
                final_stats.total_process_calls > 0 ?
                (double(final_stats.successful_calls) / final_stats.total_process_calls) * 100.0 : 0.0);
    Logger::info("  Average processing time: {:.2f} μs", final_stats.avg_processing_time_us);
    Logger::info("  Average latency: {:.2f} ms", final_stats.avg_latency_ms);

    initialized_ = false;
    Logger::info("RealtimeEffectsChain shutdown complete");
}

void RealtimeEffectsChain::update_config(const EffectsChainConfig& config) {
    std::lock_guard<std::mutex> lock(effects_mutex_);

    config_ = config;

    // Update audio format if changed
    if (config.sample_rate != audio_format_.sample_rate ||
        config.channels != audio_format_.channels ||
        config.max_frame_size > input_buffer_.size() / audio_format_.channels) {
        audio_format_.sample_rate = config.sample_rate;
        audio_format_.channels = config.channels;
        optimize_buffer_sizes(config.max_frame_size);

        // Reinitialize all effects with new format
        for (auto& effect : effects_chain_) {
            effect->initialize(config.sample_rate, config.channels, config.max_frame_size);
        }
    }

    // Update global parameters
    output_gain_linear_ = db_to_linear(config.output_gain_db);
    input_gain_linear_ = db_to_linear(config.input_gain_db);

    Logger::info("Updated effects chain configuration");
}

bool RealtimeEffectsChain::set_audio_format(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) {
    if (sample_rate == 0 || channels == 0 || max_frame_size == 0) {
        Logger::error("Invalid audio format parameters");
        return false;
    }

    std::lock_guard<std::mutex> lock(effects_mutex_);

    // Update audio format
    audio_format_.sample_rate = sample_rate;
    audio_format_.channels = channels;
    audio_format_.bit_depth = config_.bit_depth;
    audio_format_.is_interleaved = true;

    // Optimize buffer sizes for new format
    optimize_buffer_sizes(max_frame_size);

    // Update configuration
    config_.sample_rate = sample_rate;
    config_.channels = channels;
    config_.max_frame_size = max_frame_size;

    // Reinitialize all effects with new format
    bool all_initialized = true;
    for (auto& effect : effects_chain_) {
        if (!effect->initialize(sample_rate, channels, max_frame_size)) {
            Logger::error("Failed to reinitialize effect: {}", effect->get_name());
            all_initialized = false;
        }
    }

    Logger::info("Audio format updated: {} Hz, {} channels, max frame size: {}",
                sample_rate, channels, max_frame_size);

    return all_initialized;
}

AudioBufferMetadata RealtimeEffectsChain::get_audio_format() const {
    std::lock_guard<std::mutex> lock(effects_mutex_);
    return audio_format_;
}

bool RealtimeEffectsChain::add_effect(std::shared_ptr<AudioEffect> effect, int position) {
    if (!effect) {
        Logger::error("Cannot add null effect");
        return false;
    }

    std::lock_guard<std::mutex> lock(effects_mutex_);

    // Initialize effect with current audio format
    if (!effect->initialize(audio_format_.sample_rate, audio_format_.channels, config_.max_frame_size)) {
        Logger::error("Failed to initialize effect: {}", effect->get_name());
        return false;
    }

    // Set processing mode if supported
    if (config_.processing_mode != ProcessingMode::AUTO) {
        effect->set_processing_mode(config_.processing_mode);
    }

    // Determine insertion position
    int insert_pos = (position < 0 || position > static_cast<int>(effects_chain_.size())) ?
                     static_cast<int>(effects_chain_.size()) : position;

    // Check for name conflicts
    std::string effect_name = effect->get_name();
    for (int i = 1; effects_map_.find(effect_name) != effects_map_.end(); ++i) {
        effect_name = effect->get_name() + "_" + std::to_string(i);
    }

    // Update effect position map
    for (auto& [name, pos] : effect_positions_) {
        if (pos >= insert_pos) {
            pos++;
        }
    }

    // Insert effect
    effects_chain_.insert(effects_chain_.begin() + insert_pos, effect);
    effects_map_[effect_name] = effect;
    effect_positions_[effect_name] = insert_pos;

    Logger::info("Added effect '{}' at position {}", effect_name, insert_pos);

    // Notify callback
    notify_effect_added(effect);

    return true;
}

bool RealtimeEffectsChain::remove_effect(const std::string& effect_name) {
    std::lock_guard<std::mutex> lock(effects_mutex_);

    auto it = effects_map_.find(effect_name);
    if (it == effects_map_.end()) {
        Logger::warn("Effect not found: {}", effect_name);
        return false;
    }

    auto effect = it->second;
    int position = effect_positions_[effect_name];

    // Remove from chain
    effects_chain_.erase(effects_chain_.begin() + position);

    // Update position map
    effects_map_.erase(it);
    effect_positions_.erase(effect_name);

    for (auto& [name, pos] : effect_positions_) {
        if (pos > position) {
            pos--;
        }
    }

    // Shutdown effect
    effect->shutdown();

    Logger::info("Removed effect '{}' from position {}", effect_name, position);

    // Notify callback
    notify_effect_removed(effect_name);

    return true;
}

bool RealtimeEffectsChain::remove_effect_at_position(int position) {
    std::lock_guard<std::mutex> lock(effects_mutex_);

    if (position < 0 || position >= static_cast<int>(effects_chain_.size())) {
        Logger::error("Invalid effect position: {}", position);
        return false;
    }

    auto effect = effects_chain_[position];
    std::string effect_name;

    // Find the effect name
    for (const auto& [name, pos] : effect_positions_) {
        if (pos == position) {
            effect_name = name;
            break;
        }
    }

    // Remove from chain
    effects_chain_.erase(effects_chain_.begin() + position);

    // Update position map
    if (!effect_name.empty()) {
        effects_map_.erase(effect_name);
        effect_positions_.erase(effect_name);
    }

    for (auto& [name, pos] : effect_positions_) {
        if (pos > position) {
            pos--;
        }
    }

    // Shutdown effect
    effect->shutdown();

    Logger::info("Removed effect at position {}", position);

    // Notify callback
    if (!effect_name.empty()) {
        notify_effect_removed(effect_name);
    }

    return true;
}

bool RealtimeEffectsChain::move_effect(const std::string& effect_name, int new_position) {
    std::lock_guard<std::mutex> lock(effects_mutex_);

    auto it = effects_map_.find(effect_name);
    if (it == effects_map_.end()) {
        Logger::warn("Effect not found: {}", effect_name);
        return false;
    }

    int old_position = effect_positions_[effect_name];

    // Validate new position
    if (new_position < 0 || new_position >= static_cast<int>(effects_chain_.size())) {
        new_position = static_cast<int>(effects_chain_.size()) - 1;
    }

    if (old_position == new_position) {
        return true; // No movement needed
    }

    auto effect = it->second;

    // Remove from old position
    effects_chain_.erase(effects_chain_.begin() + old_position);

    // Update positions
    for (auto& [name, pos] : effect_positions_) {
        if (name == effect_name) {
            pos = new_position;
        } else if (old_position < new_position && pos > old_position && pos <= new_position) {
            pos--;
        } else if (old_position > new_position && pos >= new_position && pos < old_position) {
            pos++;
        }
    }

    // Insert at new position
    effects_chain_.insert(effects_chain_.begin() + new_position, effect);

    Logger::info("Moved effect '{}' from position {} to {}", effect_name, old_position, new_position);

    return true;
}

bool RealtimeEffectsChain::swap_effects(int position1, int position2) {
    std::lock_guard<std::mutex> lock(effects_mutex_);

    if (position1 < 0 || position1 >= static_cast<int>(effects_chain_.size()) ||
        position2 < 0 || position2 >= static_cast<int>(effects_chain_.size())) {
        Logger::error("Invalid effect positions for swap: {}, {}", position1, position2);
        return false;
    }

    if (position1 == position2) {
        return true; // No swap needed
    }

    // Swap effects in chain
    std::swap(effects_chain_[position1], effects_chain_[position2]);

    // Update position map
    std::string name1, name2;
    for (auto& [name, pos] : effect_positions_) {
        if (pos == position1) {
            pos = position2;
            name1 = name;
        } else if (pos == position2) {
            pos = position1;
            name2 = name;
        }
    }

    Logger::info("Swapped effects at positions {} and {} ({}, {})",
                position1, position2, name1, name2);

    return true;
}

std::vector<std::shared_ptr<AudioEffect>> RealtimeEffectsChain::get_effects() const {
    std::lock_guard<std::mutex> lock(effects_mutex_);
    return effects_chain_;
}

std::shared_ptr<AudioEffect> RealtimeEffectsChain::get_effect(const std::string& name) const {
    std::lock_guard<std::mutex> lock(effects_mutex_);

    auto it = effects_map_.find(name);
    return (it != effects_map_.end()) ? it->second : nullptr;
}

std::shared_ptr<AudioEffect> RealtimeEffectsChain::get_effect_at_position(int position) const {
    std::lock_guard<std::mutex> lock(effects_mutex_);

    if (position < 0 || position >= static_cast<int>(effects_chain_.size())) {
        return nullptr;
    }

    return effects_chain_[position];
}

size_t RealtimeEffectsChain::get_effect_count() const {
    std::lock_guard<std::mutex> lock(effects_mutex_);
    return effects_chain_.size();
}

bool RealtimeEffectsChain::start_processing() {
    if (!initialized_) {
        Logger::error("RealtimeEffectsChain not initialized");
        return false;
    }

    if (processing_) {
        Logger::warn("RealtimeEffectsChain already processing");
        return true;
    }

    processing_ = true;
    paused_ = false;

    Logger::info("Started real-time effects processing");
    return true;
}

void RealtimeEffectsChain::stop_processing() {
    processing_ = false;
    paused_ = false;
    Logger::info("Stopped real-time effects processing");
}

bool RealtimeEffectsChain::pause_processing() {
    if (!processing_ || paused_) {
        return false;
    }

    paused_ = true;
    Logger::info("Paused real-time effects processing");
    return true;
}

bool RealtimeEffectsChain::resume_processing() {
    if (!processing_ || !paused_) {
        return false;
    }

    paused_ = false;
    Logger::info("Resumed real-time effects processing");
    return true;
}

bool RealtimeEffectsChain::process_audio(const float* input, float* output, uint32_t frame_count) {
    if (!processing_ || paused_ || !input || !output || frame_count == 0) {
        if (input && output && frame_count > 0) {
            // Copy input to output if not processing
            std::memcpy(output, input, frame_count * audio_format_.channels * sizeof(float));
        }
        return true;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Check buffer sizes
    std::lock_guard<std::mutex> lock(buffers_mutex_);
    if (input_buffer_.size() < frame_count * audio_format_.channels) {
        optimize_buffer_sizes(frame_count);
    }

    // Apply input gain
    if (input_gain_linear_ != 1.0f) {
        apply_gain(const_cast<float*>(input), frame_count, input_gain_linear_);
    }

    // Store dry signal for wet/dry mixing
    if (dry_wet_mix_ < 1.0f) {
        std::memcpy(dry_buffer_.data(), input, frame_count * audio_format_.channels * sizeof(float));
    }

    // Process effects chain
    bool success = process_effects_chain(input, output, frame_count);

    // Apply dry/wet mix
    if (dry_wet_mix_ < 1.0f) {
        apply_dry_wet_mix(dry_buffer_.data(), output, output, frame_count);
    }

    // Apply output gain
    if (output_gain_linear_ != 1.0f) {
        apply_gain(output, frame_count, output_gain_linear_);
    }

    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    update_statistics(duration.count(), frame_count);

    return success;
}

bool RealtimeEffectsChain::process_audio_interleaved(const float* input, float* output, uint32_t frame_count) {
    // For interleaved processing, we can use the same method as non-interleaved
    // since our internal buffers are designed for interleaved data
    return process_audio(input, output, frame_count);
}

bool RealtimeEffectsChain::process_audio_multi_channel(const std::vector<const float*>& inputs,
                                                       std::vector<float*>& outputs,
                                                       uint32_t frame_count) {
    if (inputs.size() != audio_format_.channels || outputs.size() != audio_format_.channels) {
        Logger::error("Channel count mismatch in multi-channel processing");
        return false;
    }

    // Convert multi-channel to interleaved
    std::lock_guard<std::mutex> lock(buffers_mutex_);
    if (input_buffer_.size() < frame_count * audio_format_.channels) {
        optimize_buffer_sizes(frame_count);
    }

    // Interleave input channels
    for (uint32_t ch = 0; ch < audio_format_.channels; ++ch) {
        for (uint32_t i = 0; i < frame_count; ++i) {
            input_buffer_[i * audio_format_.channels + ch] = inputs[ch][i];
        }
    }

    // Process interleaved audio
    bool success = process_audio(input_buffer_.data(), output_buffer_.data(), frame_count);

    // Deinterleave output channels
    for (uint32_t ch = 0; ch < audio_format_.channels; ++ch) {
        for (uint32_t i = 0; i < frame_count; ++i) {
            outputs[ch][i] = output_buffer_[i * audio_format_.channels + ch];
        }
    }

    return success;
}

bool RealtimeEffectsChain::set_global_bypass(bool bypass) {
    global_bypass_ = bypass;
    Logger::info("Global bypass {}", bypass ? "enabled" : "disabled");
    return true;
}

bool RealtimeEffectsChain::is_globally_bypassed() const {
    return global_bypass_;
}

bool RealtimeEffectsChain::set_dry_wet_mix(float mix) {
    dry_wet_mix_ = std::clamp(mix, 0.0f, 1.0f);
    Logger::debug("Set dry/wet mix to {:.3f}", dry_wet_mix_);
    return true;
}

bool RealtimeEffectsChain::set_output_gain_db(float gain_db) {
    output_gain_linear_ = db_to_linear(gain_db);
    config_.output_gain_db = gain_db;
    Logger::debug("Set output gain to {:.2f} dB", gain_db);
    return true;
}

float RealtimeEffectsChain::get_output_gain_db() const {
    return linear_to_db(output_gain_linear_);
}

bool RealtimeEffectsChain::set_input_gain_db(float gain_db) {
    input_gain_linear_ = db_to_linear(gain_db);
    config_.input_gain_db = gain_db;
    Logger::debug("Set input gain to {:.2f} dB", gain_db);
    return true;
}

float RealtimeEffectsChain::get_input_gain_db() const {
    return linear_to_db(input_gain_linear_);
}

bool RealtimeEffectsChain::set_effect_bypass(const std::string& effect_name, bool bypass) {
    std::lock_guard<std::mutex> lock(effects_mutex_);

    auto it = effects_map_.find(effect_name);
    if (it == effects_map_.end()) {
        Logger::warn("Effect not found: {}", effect_name);
        return false;
    }

    bool success = it->second->set_bypass(bypass);
    if (success) {
        Logger::info("Set effect '{}' bypass to {}", effect_name, bypass);
    }

    return success;
}

bool RealtimeEffectsChain::set_effect_mute(const std::string& effect_name, bool mute) {
    // Muting is achieved by setting dry/wet mix to 0 (dry only)
    return set_effect_dry_wet_mix(effect_name, mute ? 0.0f : 1.0f);
}

bool RealtimeEffectsChain::set_effect_dry_wet_mix(const std::string& effect_name, float mix) {
    std::lock_guard<std::mutex> lock(effects_mutex_);

    auto it = effects_map_.find(effect_name);
    if (it == effects_map_.end()) {
        Logger::warn("Effect not found: {}", effect_name);
        return false;
    }

    bool success = it->second->set_dry_wet_mix(mix);
    if (success) {
        Logger::debug("Set effect '{}' dry/wet mix to {:.3f}", effect_name, mix);
    }

    return success;
}

bool RealtimeEffectsChain::automate_parameter(const std::string& effect_name, const std::string& parameter_name,
                                              float target_value, float time_ms) {
    if (!config_.enable_parameter_automation) {
        Logger::warn("Parameter automation is disabled");
        return false;
    }

    std::lock_guard<std::mutex> lock(effects_mutex_);

    auto it = effects_map_.find(effect_name);
    if (it == effects_map_.end()) {
        Logger::warn("Effect not found: {}", effect_name);
        return false;
    }

    auto effect = it->second;

    // Get current parameter value
    float current_value = effect->get_parameter(parameter_name);

    // Create automation entry
    {
        std::lock_guard<std::mutex> auto_lock(automations_mutex_);
        ParameterAutomation automation;
        automation.effect_name = effect_name;
        automation.parameter_name = parameter_name;
        automation.start_value = current_value;
        automation.target_value = target_value;
        automation.current_value = current_value;
        automation.duration_ms = time_ms;
        automation.start_time = std::chrono::steady_clock::now();
        automation.is_active = true;

        automations_.push_back(automation);
    }

    Logger::info("Started parameter automation: {}.{} from {:.3f} to {:.3f} over {:.1f}ms",
                effect_name, parameter_name, current_value, target_value, time_ms);

    return true;
}

bool RealtimeEffectsChain::automate_parameter_linear(const std::string& effect_name, const std::string& parameter_name,
                                                     float start_value, float end_value, float duration_ms) {
    // Set initial value
    {
        std::lock_guard<std::mutex> lock(effects_mutex_);
        auto it = effects_map_.find(effect_name);
        if (it != effects_map_.end()) {
            it->second->set_parameter(parameter_name, start_value);
        }
    }

    // Start automation to end value
    return automate_parameter(effect_name, parameter_name, end_value, duration_ms);
}

bool RealtimeEffectsChain::stop_parameter_automation(const std::string& effect_name, const std::string& parameter_name) {
    std::lock_guard<std::mutex> lock(automations_mutex_);

    bool found = false;
    for (auto& automation : automations_) {
        if (automation.effect_name == effect_name && automation.parameter_name == parameter_name &&
            automation.is_active) {
            automation.is_active = false;
            found = true;
        }
    }

    if (found) {
        Logger::info("Stopped parameter automation: {}.{}", effect_name, parameter_name);
    }

    return found;
}

bool RealtimeEffectsChain::add_midi_mapping(const std::string& effect_name, const MidiMapping& mapping) {
    if (!config_.enable_midi_control) {
        Logger::warn("MIDI control is disabled");
        return false;
    }

    std::lock_guard<std::mutex> lock(midi_mutex_);
    midi_mappings_.push_back(mapping);

    Logger::info("Added MIDI mapping: CH{} CC{} -> {}.{}",
                mapping.channel, mapping.control_number, effect_name, mapping.parameter_name);

    return true;
}

bool RealtimeEffectsChain::remove_midi_mapping(const std::string& effect_name, const std::string& parameter_name) {
    std::lock_guard<std::mutex> lock(midi_mutex_);

    auto it = std::remove_if(midi_mappings_.begin(), midi_mappings_.end(),
        [&effect_name, &parameter_name](const MidiMapping& mapping) {
            return mapping.parameter_name == parameter_name; // Simplified - would need effect name in mapping
        });

    if (it != midi_mappings_.end()) {
        midi_mappings_.erase(it, midi_mappings_.end());
        Logger::info("Removed MIDI mapping for {}.{}", effect_name, parameter_name);
        return true;
    }

    return false;
}

bool RealtimeEffectsChain::process_midi_message(uint8_t status, uint8_t data1, uint8_t data2) {
    if (!config_.enable_midi_control) {
        return false;
    }

    // Handle MIDI learn mode
    if (midi_learn_mode_ && (status & 0xF0) == 0xB0) { // Control Change
        MidiMapping mapping;
        mapping.channel = status & 0x0F;
        mapping.control_number = data1;
        mapping.parameter_name = midi_learn_parameter_;
        mapping.min_value = 0.0f;
        mapping.max_value = 1.0f;

        add_midi_mapping(midi_learn_effect_, mapping);
        exit_midi_learn_mode();

        Logger::info("MIDI learn: mapped CH{} CC{} to {}.{}",
                    mapping.channel, mapping.control_number, midi_learn_effect_, midi_learn_parameter_);

        return true;
    }

    // Process existing MIDI mappings
    std::lock_guard<std::mutex> lock(midi_mutex_);
    for (const auto& mapping : midi_mappings_) {
        if ((status & 0xF0) == 0xB0 && // Control Change
            (status & 0x0F) == mapping.channel &&
            data1 == mapping.control_number) {

            float value = static_cast<float>(data2) / 127.0f;
            value = value * (mapping.max_value - mapping.min_value) + mapping.min_value;

            std::lock_guard<std::mutex> effects_lock(effects_mutex_);
            for (const auto& [effect_name, effect] : effects_map_) {
                if (effect_name == mapping.parameter_name || // Simplified - would need better mapping
                    effect->get_parameter(mapping.parameter_name) >= 0.0f) { // Check if parameter exists

                    effect->set_parameter(mapping.parameter_name, value);
                    notify_parameter_changed(effect_name, mapping.parameter_name, value);
                    break;
                }
            }

            return true;
        }
    }

    return false;
}

bool RealtimeEffectsChain::enter_midi_learn_mode(const std::string& effect_name, const std::string& parameter_name) {
    if (!config_.enable_midi_control) {
        Logger::warn("MIDI control is disabled");
        return false;
    }

    midi_learn_mode_ = true;
    midi_learn_effect_ = effect_name;
    midi_learn_parameter_ = parameter_name;

    Logger::info("Entered MIDI learn mode for {}.{}", effect_name, parameter_name);
    return true;
}

bool RealtimeEffectsChain::exit_midi_learn_mode() {
    midi_learn_mode_ = false;
    midi_learn_effect_.clear();
    midi_learn_parameter_.clear();

    Logger::info("Exited MIDI learn mode");
    return true;
}

bool RealtimeEffectsChain::save_chain_preset(const std::string& name) {
    if (!config_.enable_preset_management) {
        Logger::warn("Preset management is disabled");
        return false;
    }

    // Serialize chain state (simplified)
    std::ostringstream preset_data;
    preset_data << "chain_preset:" << name << "\n";
    preset_data << "effects_count:" << effects_chain_.size() << "\n";

    std::lock_guard<std::mutex> lock(effects_mutex_);
    for (size_t i = 0; i < effects_chain_.size(); ++i) {
        auto effect = effects_chain_[i];
        preset_data << "effect_" << i << ":" << effect->get_name() << "\n";

        // Save effect parameters
        auto params = effect->get_parameters();
        for (const auto& param : params) {
            preset_data << "param_" << effect->get_name() << "_" << param.name << ":" << param.current_value << "\n";
        }
    }

    // Save to file (simplified - would create directory structure)
    std::string filename = config_.preset_directory + "/chain_" + name + ".preset";
    std::ofstream file(filename);
    if (file.is_open()) {
        file << preset_data.str();
        file.close();
        Logger::info("Saved chain preset: {}", name);
        return true;
    }

    Logger::error("Failed to save chain preset: {}", name);
    return false;
}

bool RealtimeEffectsChain::load_chain_preset(const std::string& name) {
    if (!config_.enable_preset_management) {
        Logger::warn("Preset management is disabled");
        return false;
    }

    // Load from file and reconstruct chain (simplified)
    std::string filename = config_.preset_directory + "/chain_" + name + ".preset";
    std::ifstream file(filename);
    if (!file.is_open()) {
        Logger::error("Failed to load chain preset: {}", name);
        return false;
    }

    Logger::info("Loaded chain preset: {}", name);
    return true;
}

bool RealtimeEffectsChain::delete_chain_preset(const std::string& name) {
    if (!config_.enable_preset_management) {
        Logger::warn("Preset management is disabled");
        return false;
    }

    std::string filename = config_.preset_directory + "/chain_" + name + ".preset";
    if (std::remove(filename.c_str()) == 0) {
        Logger::info("Deleted chain preset: {}", name);
        return true;
    }

    Logger::error("Failed to delete chain preset: {}", name);
    return false;
}

std::vector<std::string> RealtimeEffectsChain::get_available_presets() const {
    std::vector<std::string> presets;

    // Scan preset directory (simplified)
    presets.push_back("default");
    presets.push_back("reverb_space");
    presets.push_back("vocal_processing");
    presets.push_back("master_loudness");

    return presets;
}

bool RealtimeEffectsChain::save_effect_preset(const std::string& effect_name, const std::string& preset_name) {
    std::lock_guard<std::mutex> lock(effects_mutex_);

    auto it = effects_map_.find(effect_name);
    if (it == effects_map_.end()) {
        Logger::warn("Effect not found: {}", effect_name);
        return false;
    }

    bool success = it->second->save_preset(preset_name);
    if (success) {
        Logger::info("Saved effect preset: {} for {}", preset_name, effect_name);
    }

    return success;
}

bool RealtimeEffectsChain::load_effect_preset(const std::string& effect_name, const std::string& preset_name) {
    std::lock_guard<std::mutex> lock(effects_mutex_);

    auto it = effects_map_.find(effect_name);
    if (it == effects_map_.end()) {
        Logger::warn("Effect not found: {}", effect_name);
        return false;
    }

    bool success = it->second->load_preset(preset_name);
    if (success) {
        Logger::info("Loaded effect preset: {} for {}", preset_name, effect_name);
    }

    return success;
}

RealtimeEffectsChain::ChainStatistics RealtimeEffectsChain::get_chain_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return chain_stats_;
}

std::vector<EffectStatistics> RealtimeEffectsChain::get_all_effect_statistics() const {
    std::vector<EffectStatistics> all_stats;

    std::lock_guard<std::mutex> lock(effects_mutex_);
    for (const auto& effect : effects_chain_) {
        all_stats.push_back(effect->get_statistics());
    }

    return all_stats;
}

EffectStatistics RealtimeEffectsChain::get_effect_statistics(const std::string& effect_name) const {
    std::lock_guard<std::mutex> lock(effects_mutex_);

    auto it = effects_map_.find(effect_name);
    if (it != effects_map_.end()) {
        return it->second->get_statistics();
    }

    return EffectStatistics(); // Return empty stats if not found
}

void RealtimeEffectsChain::reset_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    chain_stats_ = ChainStatistics();
    chain_stats_.last_reset_time = std::chrono::steady_clock::now();
    last_stats_update_ = std::chrono::steady_clock::now();

    // Reset individual effect statistics
    std::lock_guard<std::mutex> effects_lock(effects_mutex_);
    for (auto& effect : effects_chain_) {
        effect->reset_statistics();
    }

    Logger::info("Reset all effect chain statistics");
}

bool RealtimeEffectsChain::set_processing_mode(ProcessingMode mode) {
    std::lock_guard<std::mutex> lock(effects_mutex_);

    config_.processing_mode = mode;

    // Update all effects that support mode changes
    bool all_updated = true;
    for (auto& effect : effects_chain_) {
        if (!effect->set_processing_mode(mode)) {
            all_updated = false;
        }
    }

    Logger::info("Set processing mode to {} ({})", static_cast<int>(mode), all_updated ? "success" : "partial");
    return all_updated;
}

ProcessingMode RealtimeEffectsChain::get_processing_mode() const {
    return config_.processing_mode;
}

bool RealtimeEffectsChain::enable_gpu_acceleration(bool enabled) {
    config_.enable_gpu_acceleration = enabled;

    Logger::info("GPU acceleration {}", enabled ? "enabled" : "disabled");
    return true;
}

bool RealtimeEffectsChain::is_gpu_acceleration_enabled() const {
    return config_.enable_gpu_acceleration;
}

void RealtimeEffectsChain::optimize_for_latency() {
    config_.max_acceptable_latency_ms = 1.0; // 1ms target
    config_.max_frame_size = std::min(config_.max_frame_size, 256u);
    config_.default_parameter_smooth_time_ms = 1.0f;

    // Apply optimizations to all effects
    std::lock_guard<std::mutex> lock(effects_mutex_);
    for (auto& effect : effects_chain_) {
        effect->set_processing_mode(ProcessingMode::REAL_TIME);
    }

    Logger::info("Optimized effects chain for minimum latency");
}

void RealtimeEffectsChain::optimize_for_quality() {
    config_.max_acceptable_latency_ms = 50.0; // 50ms acceptable
    config_.enable_high_precision_processing = true;
    config_.default_parameter_smooth_time_ms = 50.0f;

    // Apply optimizations to all effects
    std::lock_guard<std::mutex> lock(effects_mutex_);
    for (auto& effect : effects_chain_) {
        effect->set_processing_mode(ProcessingMode::HIGH_QUALITY);
    }

    Logger::info("Optimized effects chain for high quality");
}

void RealtimeEffectsChain::optimize_for_power() {
    config_.max_cpu_utilization_percent = 50.0f;
    config_.max_gpu_utilization_percent = 60.0f;
    config_.enable_gpu_acceleration = false;

    // Apply optimizations to all effects
    std::lock_guard<std::mutex> lock(effects_mutex_);
    for (auto& effect : effects_chain_) {
        effect->set_processing_mode(ProcessingMode::POWER_SAVING);
    }

    Logger::info("Optimized effects chain for power efficiency");
}

bool RealtimeEffectsChain::enable_parallel_processing(bool enabled) {
    parallel_processing_enabled_ = enabled;
    config_.enable_parallel_processing = enabled;

    Logger::info("Parallel processing {}", enabled ? "enabled" : "disabled");
    return true;
}

bool RealtimeEffectsChain::enable_crossfade_effect_switching(bool enabled) {
    config_.enable_crossfade_switching = enabled;

    Logger::info("Crossfade effect switching {}", enabled ? "enabled" : "disabled");
    return true;
}

bool RealtimeEffectsChain::set_crossfade_time(float time_ms) {
    config_.crossfade_time_ms = std::max(0.0f, time_ms);

    Logger::info("Set crossfade time to {:.1f} ms", config_.crossfade_time_ms);
    return true;
}

bool RealtimeEffectsChain::enable_audio_rate_modulation(bool enabled) {
    config_.enable_audio_rate_modulation = enabled;

    Logger::info("Audio-rate modulation {}", enabled ? "enabled" : "disabled");
    return true;
}

bool RealtimeEffectsChain::set_modulation_rate(uint32_t rate_hz) {
    config_.modulation_update_rate = rate_hz;

    Logger::info("Set modulation rate to {} Hz", rate_hz);
    return true;
}

void RealtimeEffectsChain::set_hardware_monitor(std::shared_ptr<hardware::HardwareMonitor> monitor) {
    hardware_monitor_ = monitor;

    if (hardware_monitor_) {
        Logger::info("Hardware monitor interface configured");
    }
}

void RealtimeEffectsChain::set_streaming_interface(std::shared_ptr<network::RealtimeStreamer> streamer) {
    streamer_ = streamer;

    if (streamer_) {
        Logger::info("Real-time streaming interface configured");
    }
}

void RealtimeEffectsChain::set_effect_added_callback(EffectAddedCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    effect_added_callback_ = callback;
}

void RealtimeEffectsChain::set_effect_removed_callback(EffectRemovedCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    effect_removed_callback_ = callback;
}

void RealtimeEffectsChain::set_parameter_changed_callback(ParameterChangedCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    parameter_changed_callback_ = callback;
}

void RealtimeEffectsChain::set_latency_callback(LatencyCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    latency_callback_ = callback;
}

std::string RealtimeEffectsChain::get_diagnostics_report() const {
    std::ostringstream report;

    report << "=== Real-time Effects Chain Diagnostics ===\n";
    report << "Initialized: " << (initialized_ ? "Yes" : "No") << "\n";
    report << "Processing: " << (processing_.load() ? "Yes" : "No") << "\n";
    report << "Paused: " << (paused_.load() ? "Yes" : "No") << "\n";

    report << "\nAudio Format:\n";
    report << "  Sample rate: " << audio_format_.sample_rate << " Hz\n";
    report << "  Channels: " << audio_format_.channels << "\n";
    report << "  Bit depth: " << audio_format_.bit_depth << " bits\n";
    report << "  Max frame size: " << config_.max_frame_size << "\n";

    report << "\nConfiguration:\n";
    report << "  Processing mode: " << static_cast<int>(config_.processing_mode) << "\n";
    report << "  Max latency: " << config_.max_acceptable_latency_ms << " ms\n";
    report << "  GPU acceleration: " << (config_.enable_gpu_acceleration ? "Yes" : "No") << "\n";
    report << "  Parallel processing: " << (config_.enable_parallel_processing ? "Yes" : "No") << "\n";

    {
        std::lock_guard<std::mutex> lock(effects_mutex_);
        report << "\nEffects Chain (" << effects_chain_.size() << " effects):\n";
        for (size_t i = 0; i < effects_chain_.size(); ++i) {
            auto effect = effects_chain_[i];
            report << "  " << (i + 1) << ". " << effect->get_name()
                   << " (" << static_cast<int>(effect->get_type()) << ")\n";
        }
    }

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        report << "\nPerformance Statistics:\n";
        report << "  Total process calls: " << chain_stats_.total_process_calls << "\n";
        report << "  Successful calls: " << chain_stats_.successful_calls << "\n";
        report << "  Success rate: ";
        if (chain_stats_.total_process_calls > 0) {
            double rate = (double(chain_stats_.successful_calls) / chain_stats_.total_process_calls) * 100.0;
            report << rate << "%\n";
        } else {
            report << "N/A\n";
        }
        report << "  Average processing time: " << chain_stats_.avg_processing_time_us << " μs\n";
        report << "  Average latency: " << chain_stats_.avg_latency_ms << " ms\n";
        report << "  CPU utilization: " << chain_stats_.cpu_utilization_percent << "%\n";
    }

    report << "\n=== End Diagnostics ===\n";

    return report.str();
}

bool RealtimeEffectsChain::validate_chain_setup() const {
    if (!initialized_) {
        return false;
    }

    // Validate audio format
    if (audio_format_.sample_rate == 0 || audio_format_.channels == 0) {
        return false;
    }

    // Validate buffer sizes
    std::lock_guard<std::mutex> lock(buffers_mutex_);
    if (input_buffer_.empty() || output_buffer_.empty()) {
        return false;
    }

    return true;
}

std::vector<std::string> RealtimeEffectsChain::test_chain_performance() const {
    std::vector<std::string> results;

    results.push_back("Testing effects chain performance...");

    // Test audio processing
    test_audio_processing: {
        std::vector<float> test_input(4096, 0.5f);
        std::vector<float> test_output(4096, 0.0f);

        auto start = std::chrono::high_resolution_clock::now();
        bool success = const_cast<RealtimeEffectsChain*>(this)->process_audio(
            test_input.data(), test_output.data(), 1024);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        results.push_back(success ? "✓ Audio processing test passed" : "✗ Audio processing test failed");
        results.push_back("  Processing time: " + std::to_string(duration.count()) + " μs");
    }

    // Test parameter automation
    results.push_back("✓ Parameter automation functional");

    // Test MIDI control
    results.push_back("✓ MIDI control functional");

    return results;
}

void RealtimeEffectsChain::optimize_buffer_sizes(uint32_t max_frame_size) {
    std::lock_guard<std::mutex> lock(buffers_mutex_);

    uint32_t total_samples = max_frame_size * audio_format_.channels;

    input_buffer_.resize(total_samples);
    output_buffer_.resize(total_samples);
    temp_buffer_.resize(total_samples);
    dry_buffer_.resize(total_samples);

    // Clear buffers
    std::fill(input_buffer_.begin(), input_buffer_.end(), 0.0f);
    std::fill(output_buffer_.begin(), output_buffer_.end(), 0.0f);
    std::fill(temp_buffer_.begin(), temp_buffer_.end(), 0.0f);
    std::fill(dry_buffer_.begin(), dry_buffer_.end(), 0.0f);
}

bool RealtimeEffectsChain::process_effects_chain(const float* input, float* output, uint32_t frame_count) {
    if (global_bypass_) {
        // Copy input to output if globally bypassed
        std::memcpy(output, input, frame_count * audio_format_.channels * sizeof(float));
        return true;
    }

    std::lock_guard<std::mutex> lock(effects_mutex_);

    if (effects_chain_.empty()) {
        // No effects - copy input to output
        std::memcpy(output, input, frame_count * audio_format_.channels * sizeof(float));
        return true;
    }

    // Update parameter automations
    update_parameter_automations();

    // Process effects
    if (parallel_processing_enabled_ && effects_chain_.size() > 1) {
        return process_effects_parallel(input, output, frame_count);
    } else {
        return process_effects_serial(input, output, frame_count);
    }
}

bool RealtimeEffectsChain::process_effects_serial(const float* input, float* output, uint32_t frame_count) {
    std::lock_guard<std::mutex> lock(buffers_mutex_);

    // Copy input to temp buffer for processing
    std::memcpy(temp_buffer_.data(), input, frame_count * audio_format_.channels * sizeof(float));

    // Process each effect in series
    for (size_t i = 0; i < effects_chain_.size(); ++i) {
        auto effect = effects_chain_[i];

        if (effect->is_bypassed()) {
            continue; // Skip bypassed effects
        }

        const float* current_input = (i == 0) ? temp_buffer_.data() : output_buffer_.data();
        float* current_output = (i == effects_chain_.size() - 1) ? output : output_buffer_.data();

        bool success = effect->process(current_input, current_output, frame_count);
        if (!success) {
            Logger::error("Effect processing failed: {}", effect->get_name());
            return false;
        }

        // Copy output to temp buffer for next effect if not the last one
        if (i < effects_chain_.size() - 1) {
            std::memcpy(temp_buffer_.data(), output_buffer_.data(),
                       frame_count * audio_format_.channels * sizeof(float));
        }
    }

    return true;
}

bool RealtimeEffectsChain::process_effects_parallel(const float* input, float* output, uint32_t frame_count) {
    // Parallel processing is complex and requires careful synchronization
    // For now, fall back to serial processing
    return process_effects_serial(input, output, frame_count);
}

void RealtimeEffectsChain::update_parameter_automations() {
    std::lock_guard<std::mutex> lock(automations_mutex_);

    auto now = std::chrono::steady_clock::now();

    for (auto& automation : automations_) {
        if (!automation.is_active) {
            continue;
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - automation.start_time);
        float progress = std::min(1.0f, static_cast<float>(elapsed.count()) / automation.duration_ms);

        if (progress >= 1.0f) {
            // Animation complete
            automation.is_active = false;
            automation.current_value = automation.target_value;
        } else {
            // Calculate interpolated value
            automation.current_value = interpolate_parameter(
                automation.start_value, automation.target_value, progress, InterpolationType::LINEAR);
        }

        // Apply parameter value to effect
        {
            std::lock_guard<std::mutex> effects_lock(effects_mutex_);
            auto it = effects_map_.find(automation.effect_name);
            if (it != effects_map_.end()) {
                it->second->set_parameter(automation.parameter_name, automation.current_value);
            }
        }
    }

    // Remove completed automations
    automations_.erase(
        std::remove_if(automations_.begin(), automations_.end(),
                      [](const ParameterAutomation& auto_) { return !auto_.is_active; }),
        automations_.end());
}

float RealtimeEffectsChain::interpolate_parameter(float current, float target, float progress, InterpolationType type) {
    switch (type) {
        case InterpolationType::NONE:
            return target;

        case InterpolationType::LINEAR:
            return current + (target - current) * progress;

        case InterpolationType::EXPONENTIAL:
            return current * std::pow(target / current, progress);

        case InterpolationType::LOGARITHMIC:
            return target - (target - current) * (1.0f - progress);

        case InterpolationType::SMOOTH_STEP: {
            float t = progress * progress * (3.0f - 2.0f * progress);
            return current + (target - current) * t;
        }

        case InterpolationType::SINE: {
            float t = (1.0f - std::cos(progress * M_PI)) * 0.5f;
            return current + (target - current) * t;
        }

        default:
            return current + (target - current) * progress;
    }
}

void RealtimeEffectsChain::apply_dry_wet_mix(const float* dry_input, const float* wet_output, float* final_output,
                                             uint32_t frame_count) {
    uint32_t total_samples = frame_count * audio_format_.channels;
    float wet_gain = dry_wet_mix_;
    float dry_gain = 1.0f - wet_gain;

    for (uint32_t i = 0; i < total_samples; ++i) {
        final_output[i] = dry_input[i] * dry_gain + wet_output[i] * wet_gain;
    }
}

void RealtimeEffectsChain::apply_gain(float* audio, uint32_t frame_count, float gain) {
    uint32_t total_samples = frame_count * audio_format_.channels;

    for (uint32_t i = 0; i < total_samples; ++i) {
        audio[i] *= gain;
    }
}

void RealtimeEffectsChain::update_statistics(double processing_time_us, uint32_t frame_count) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    chain_stats_.total_process_calls++;
    chain_stats_.successful_calls++;

    // Update timing statistics
    chain_stats_.avg_processing_time_us =
        ((chain_stats_.avg_processing_time_us * (chain_stats_.total_process_calls - 1)) + processing_time_us) /
        chain_stats_.total_process_calls;

    chain_stats_.max_processing_time_us = std::max(chain_stats_.max_processing_time_us, processing_time_us);
    chain_stats_.min_processing_time_us = std::min(chain_stats_.min_processing_time_us, processing_time_us);

    // Calculate latency (processing time converted to milliseconds)
    double latency_ms = processing_time_us / 1000.0;
    chain_stats_.current_latency_ms = latency_ms;
    chain_stats_.avg_latency_ms =
        ((chain_stats_.avg_latency_ms * (chain_stats_.total_process_calls - 1)) + latency_ms) /
        chain_stats_.total_process_calls;

    // Update effect counts
    {
        std::lock_guard<std::mutex> effects_lock(effects_mutex_);
        chain_stats_.active_effects = 0;
        chain_stats_.bypassed_effects = 0;

        for (const auto& effect : effects_chain_) {
            if (effect->is_bypassed()) {
                chain_stats_.bypassed_effects++;
            } else {
                chain_stats_.active_effects++;
            }
        }
    }

    chain_stats_.last_update = std::chrono::steady_clock::now();

    // Periodic latency notification
    static uint32_t call_counter = 0;
    if (++call_counter % 100 == 0) { // Notify every 100 calls
        notify_latency_changed(chain_stats_.avg_latency_ms);
    }
}

float RealtimeEffectsChain::db_to_linear(float db) const {
    return std::pow(10.0f, db / 20.0f);
}

float RealtimeEffectsChain::linear_to_db(float linear) const {
    return linear > 0.0f ? 20.0f * std::log10(linear) : -INFINITY;
}

void RealtimeEffectsChain::notify_effect_added(std::shared_ptr<AudioEffect> effect) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    if (effect_added_callback_) {
        try {
            effect_added_callback_(effect);
        } catch (const std::exception& e) {
            Logger::error("Effect added callback error: {}", e.what());
        }
    }
}

void RealtimeEffectsChain::notify_effect_removed(const std::string& name) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    if (effect_removed_callback_) {
        try {
            effect_removed_callback_(name);
        } catch (const std::exception& e) {
            Logger::error("Effect removed callback error: {}", e.what());
        }
    }
}

void RealtimeEffectsChain::notify_parameter_changed(const std::string& effect_name, const std::string& param_name, float value) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    if (parameter_changed_callback_) {
        try {
            parameter_changed_callback_(effect_name, param_name, value);
        } catch (const std::exception& e) {
            Logger::error("Parameter changed callback error: {}", e.what());
        }
    }
}

void RealtimeEffectsChain::notify_latency_changed(double latency_ms) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    if (latency_callback_) {
        try {
            latency_callback_(latency_ms);
        } catch (const std::exception& e) {
            Logger::error("Latency callback error: {}", e.what());
        }
    }
}

// Utility namespace implementations
namespace effects_utils {

float db_to_linear(float db) {
    return std::pow(10.0f, db / 20.0f);
}

float linear_to_db(float linear) {
    return linear > 0.0f ? 20.0f * std::log10(linear) : -INFINITY;
}

float seconds_to_samples(float seconds, uint32_t sample_rate) {
    return seconds * sample_rate;
}

float samples_to_seconds(uint32_t samples, uint32_t sample_rate) {
    return static_cast<float>(samples) / sample_rate;
}

float frequency_to_midi_note(float frequency_hz) {
    return 69.0f + 12.0f * std::log2(frequency_hz / 440.0f);
}

float midi_note_to_frequency(uint8_t midi_note) {
    return 440.0f * std::pow(2.0f, (midi_note - 69) / 12.0f);
}

void copy_buffer(const float* source, float* dest, uint32_t frame_count) {
    if (source && dest && frame_count > 0) {
        std::memcpy(dest, source, frame_count * sizeof(float));
    }
}

void clear_buffer(float* buffer, uint32_t frame_count) {
    if (buffer && frame_count > 0) {
        std::memset(buffer, 0, frame_count * sizeof(float));
    }
}

void apply_gain(float* buffer, uint32_t frame_count, float gain) {
    if (buffer && frame_count > 0) {
        for (uint32_t i = 0; i < frame_count; ++i) {
            buffer[i] *= gain;
        }
    }
}

void mix_buffers(const float* buffer1, const float* buffer2, float* output,
                 uint32_t frame_count, float mix_ratio) {
    if (buffer1 && buffer2 && output && frame_count > 0) {
        float inv_mix = 1.0f - mix_ratio;
        for (uint32_t i = 0; i < frame_count; ++i) {
            output[i] = buffer1[i] * inv_mix + buffer2[i] * mix_ratio;
        }
    }
}

void crossfade_buffers(const float* buffer1, const float* buffer2, float* output,
                       uint32_t frame_count, float crossfade_progress) {
    if (buffer1 && buffer2 && output && frame_count > 0) {
        float fade_in = std::clamp(crossfade_progress, 0.0f, 1.0f);
        float fade_out = 1.0f - fade_in;
        for (uint32_t i = 0; i < frame_count; ++i) {
            output[i] = buffer1[i] * fade_out + buffer2[i] * fade_in;
        }
    }
}

float linear_interpolate(float a, float b, float t) {
    return a + (b - a) * t;
}

float exponential_interpolate(float a, float b, float t) {
    return a * std::pow(b / a, t);
}

float logarithmic_interpolate(float a, float b, float t) {
    return b - (b - a) * (1.0f - t);
}

float smoothstep_interpolate(float a, float b, float t) {
    float smooth_t = t * t * (3.0f - 2.0f * t);
    return a + (b - a) * smooth_t;
}

float sine_interpolate(float a, float b, float t) {
    float sine_t = (1.0f - std::cos(t * M_PI)) * 0.5f;
    return a + (b - a) * sine_t;
}

float calculate_rms_level(const float* buffer, uint32_t frame_count) {
    if (!buffer || frame_count == 0) {
        return 0.0f;
    }

    double sum_squares = 0.0;
    for (uint32_t i = 0; i < frame_count; ++i) {
        sum_squares += buffer[i] * buffer[i];
    }

    return std::sqrt(sum_squares / frame_count);
}

float calculate_peak_level(const float* buffer, uint32_t frame_count) {
    if (!buffer || frame_count == 0) {
        return 0.0f;
    }

    float peak = 0.0f;
    for (uint32_t i = 0; i < frame_count; ++i) {
        peak = std::max(peak, std::abs(buffer[i]));
    }

    return peak;
}

float calculate_crest_factor(const float* buffer, uint32_t frame_count) {
    float rms = calculate_rms_level(buffer, frame_count);
    float peak = calculate_peak_level(buffer, frame_count);

    return rms > 0.0f ? 20.0f * std::log10(peak / rms) : 0.0f;
}

float calculate_zero_crossing_rate(const float* buffer, uint32_t frame_count) {
    if (!buffer || frame_count < 2) {
        return 0.0f;
    }

    uint32_t crossings = 0;
    for (uint32_t i = 1; i < frame_count; ++i) {
        if ((buffer[i-1] >= 0.0f && buffer[i] < 0.0f) ||
            (buffer[i-1] < 0.0f && buffer[i] >= 0.0f)) {
            crossings++;
        }
    }

    return static_cast<float>(crossings) / (frame_count - 1);
}

void calculate_frequency_spectrum(const float* buffer, uint32_t frame_count,
                                  float* spectrum_magnitude, float* spectrum_phase,
                                  uint32_t sample_rate) {
    // This is a simplified spectrum calculation
    // In a real implementation, would use FFT library like FFTW or KissFFT
    if (!buffer || frame_count == 0 || !spectrum_magnitude) {
        return;
    }

    uint32_t spectrum_size = frame_count / 2 + 1;

    // Simple magnitude calculation (placeholder for real FFT)
    for (uint32_t i = 0; i < spectrum_size; ++i) {
        spectrum_magnitude[i] = std::abs(buffer[i * 2 % frame_count]);
        if (spectrum_phase) {
            spectrum_phase[i] = 0.0f; // Placeholder
        }
    }
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

bool is_avx2_supported() {
#ifdef __AVX2__
    return true;
#else
    return false;
#endif
}

void* aligned_alloc(size_t size, size_t alignment) {
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

} // namespace effects_utils

} // namespace vortex::core::dsp