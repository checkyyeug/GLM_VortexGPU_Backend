#include "core/dsp/frequency_domain_effects.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <random>
#include <immintrin.h>

namespace vortex::core::dsp {

// FFTProcessor implementation
FFTProcessor::FFTProcessor()
    : fft_size_(0)
    , num_bins_(0)
    , channels_(0)
    , overlap_factor_(4)
    , window_type_(FFTWindowType::HANNING)
    , gpu_accelerated_(false)
    , fft_plan_(nullptr)
    , ifft_plan_(nullptr)
    , gpu_fft_context_(nullptr) {
}

FFTProcessor::~FFTProcessor() {
    shutdown();
}

bool FFTProcessor::initialize(uint32_t fft_size, uint32_t channels, bool gpu_acceleration) {
    if (!frequency_domain_utils::is_power_of_two(fft_size)) {
        Logger::error("FFT size must be power of 2");
        return false;
    }

    fft_size_ = fft_size;
    num_bins_ = fft_size / 2 + 1;
    channels_ = channels;
    gpu_accelerated_ = gpu_acceleration;

    Logger::info("Initializing FFT Processor: {} size, {} channels, GPU: {}",
                fft_size_, channels_, gpu_accelerated_ ? "Yes" : "No");

    // Initialize processing buffers
    window_buffer_.resize(fft_size_, 0.0f);
    fft_buffer_.resize(fft_size_);
    ifft_buffer_.resize(fft_size_);
    overlap_buffer_.resize(fft_size_ * channels_, 0.0f);
    input_history_.resize(fft_size_ * channels_, 0.0f);
    output_history_.resize(fft_size_ * channels_, 0.0f);

    // Generate window function
    generate_window(window_buffer_.data(), fft_size_, window_type_);

    // Create FFT plans
    if (!create_fft_plans()) {
        Logger::error("Failed to create FFT plans");
        return false;
    }

    // Initialize GPU FFT if requested
    if (gpu_accelerated_) {
        if (!initialize_gpu_fft()) {
            Logger::warn("GPU FFT initialization failed, falling back to CPU");
            gpu_accelerated_ = false;
        }
    }

    Logger::info("FFT Processor initialized successfully");
    return true;
}

void FFTProcessor::shutdown() {
    destroy_fft_plans();
    shutdown_gpu_fft();

    window_buffer_.clear();
    fft_buffer_.clear();
    ifft_buffer_.clear();
    overlap_buffer_.clear();
    input_history_.clear();
    output_history_.clear();

    fft_size_ = 0;
    num_bins_ = 0;
    channels_ = 0;

    Logger::info("FFT Processor shutdown");
}

void FFTProcessor::clear() {
    std::fill(overlap_buffer_.begin(), overlap_buffer_.end(), 0.0f);
    std::fill(input_history_.begin(), input_history_.end(), 0.0f);
    std::fill(output_history_.begin(), output_history_.end(), 0.0f);
}

bool FFTProcessor::forward_fft(const float* input, std::complex<float>* output, uint32_t frame_size) {
    if (!input || !output || frame_size > fft_size_) {
        return false;
    }

    // Apply window function
    std::vector<float> windowed_input(fft_size_, 0.0f);
    std::memcpy(windowed_input.data(), input, frame_size * sizeof(float));
    apply_window(windowed_input.data(), fft_size_, window_type_);

    // Pad with zeros if needed
    if (frame_size < fft_size_) {
        std::fill(windowed_input.begin() + frame_size, windowed_input.end(), 0.0f);
    }

    // Perform FFT
    if (gpu_accelerated_) {
        return perform_gpu_fft(windowed_input.data(), output);
    } else {
        perform_cpu_fft(windowed_input.data(), output);
        return true;
    }
}

bool FFTProcessor::inverse_fft(const std::complex<float>* input, float* output) {
    if (!input || !output) {
        return false;
    }

    if (gpu_accelerated_) {
        return perform_gpu_ifft(input, output);
    } else {
        perform_cpu_ifft(input, output);
        return true;
    }
}

bool FFTProcessor::process_overlap_add(const float* input, float* output, uint32_t frame_size) {
    if (!input || !output || frame_size > fft_size_) {
        return false;
    }

    // Apply window to current frame
    std::vector<float> windowed_input(fft_size_, 0.0f);
    std::memcpy(windowed_input.data(), input, frame_size * sizeof(float));
    apply_window(windowed_input.data(), fft_size_, window_type_);

    // Zero pad if needed
    if (frame_size < fft_size_) {
        std::fill(windowed_input.begin() + frame_size, windowed_input.end(), 0.0f);
    }

    // FFT
    std::vector<std::complex<float>> spectrum(num_bins_);
    perform_cpu_fft(windowed_input.data(), spectrum.data());

    // Process spectrum (placeholder for actual processing)
    // In a real implementation, this would apply the frequency-domain effect

    // IFFT
    std::vector<float> time_output(fft_size_);
    perform_cpu_ifft(spectrum.data(), time_output.data());

    // Apply window again and add to overlap buffer
    apply_window(time_output.data(), fft_size_, window_type_);

    // Overlap-add
    for (uint32_t i = 0; i < frame_size && i < fft_size_; ++i) {
        output[i] = time_output[i] + overlap_buffer_[i];
    }

    // Update overlap buffer
    uint32_t hop_size = fft_size_ / overlap_factor_;
    for (uint32_t i = 0; i < fft_size_ - hop_size; ++i) {
        overlap_buffer_[i] = overlap_buffer_[i + hop_size];
    }
    for (uint32_t i = fft_size_ - hop_size; i < fft_size_; ++i) {
        overlap_buffer_[i] = time_output[i];
    }

    return true;
}

void FFTProcessor::generate_window(float* window, uint32_t size, FFTWindowType window_type) {
    switch (window_type) {
        case FFTWindowType::HANNING:
            frequency_domain_utils::generate_hanning_window(window, size);
            break;
        case FFTWindowType::HAMMING:
            frequency_domain_utils::generate_hamming_window(window, size);
            break;
        case FFTWindowType::BLACKMAN:
            frequency_domain_utils::generate_blackman_window(window, size);
            break;
        case FFTWindowType::KAISER:
            frequency_domain_utils::generate_kaiser_window(window, size, 6.0f);
            break;
        default:
            frequency_domain_utils::generate_hanning_window(window, size);
            break;
    }
}

void FFTProcessor::set_fft_size(uint32_t fft_size) {
    if (frequency_domain_utils::is_power_of_two(fft_size) && fft_size != fft_size_) {
        Logger::info("Changing FFT size from {} to {}", fft_size_, fft_size);
        shutdown();
        initialize(fft_size, channels_, gpu_accelerated_);
    }
}

void FFTProcessor::set_window_type(FFTWindowType window_type) {
    window_type_ = window_type;
    generate_window(window_buffer_.data(), fft_size_, window_type_);
}

void FFTProcessor::set_overlap_factor(uint32_t overlap_factor) {
    if (overlap_factor >= 1 && overlap_factor <= 8) {
        overlap_factor_ = overlap_factor;
    }
}

bool FFTProcessor::create_fft_plans() {
    // In a real implementation, would use FFTW, Intel MKL, or similar FFT library
    // For this demo, we'll simulate plan creation
    fft_plan_ = reinterpret_cast<void*>(1); // Non-null indicates success
    ifft_plan_ = reinterpret_cast<void*>(1); // Non-null indicates success
    return true;
}

void FFTProcessor::destroy_fft_plans() {
    fft_plan_ = nullptr;
    ifft_plan_ = nullptr;
}

bool FFTProcessor::initialize_gpu_fft() {
    // In a real implementation, would initialize GPU FFT (cuFFT, clFFT, etc.)
    // For this demo, we'll simulate GPU initialization
    Logger::info("Initializing GPU FFT acceleration");
    gpu_fft_context_ = reinterpret_cast<void*>(1); // Non-null indicates success
    return true;
}

void FFTProcessor::shutdown_gpu_fft() {
    if (gpu_fft_context_) {
        // In a real implementation, would cleanup GPU FFT resources
        gpu_fft_context_ = nullptr;
    }
}

void FFTProcessor::perform_cpu_fft(const float* input, std::complex<float>* output) {
    // Simplified FFT implementation for demonstration
    // In a real implementation, would use FFTW or similar library

    std::vector<float> real_input(fft_size_);
    std::vector<float> imag_input(fft_size_, 0.0f);
    std::vector<float> real_output(fft_size_);
    std::vector<float> imag_output(fft_size_);

    // Copy input
    for (uint32_t i = 0; i < fft_size_; ++i) {
        real_input[i] = input[i];
    }

    // Simple DFT implementation (very inefficient - for demo only)
    for (uint32_t k = 0; k < num_bins_; ++k) {
        float sum_real = 0.0f;
        float sum_imag = 0.0f;

        for (uint32_t n = 0; n < fft_size_; ++n) {
            float angle = -2.0f * M_PI * k * n / fft_size_;
            sum_real += real_input[n] * std::cos(angle) - imag_input[n] * std::sin(angle);
            sum_imag += real_input[n] * std::sin(angle) + imag_input[n] * std::cos(angle);
        }

        output[k] = std::complex<float>(sum_real, sum_imag);
    }
}

void FFTProcessor::perform_cpu_ifft(const std::complex<float>* input, float* output) {
    // Simplified IFFT implementation for demonstration
    std::vector<float> real_input(fft_size_);
    std::vector<float> imag_input(fft_size_);
    std::vector<float> real_output(fft_size_);
    std::vector<float> imag_output(fft_size_);

    // Copy input and pad if needed (only real part is needed for real signals)
    for (uint32_t i = 0; i < num_bins_; ++i) {
        real_input[i] = input[i].real();
        imag_input[i] = input[i].imag();
    }
    // Mirror for full spectrum
    for (uint32_t i = num_bins_; i < fft_size_; ++i) {
        uint32_t mirror_idx = fft_size_ - i;
        real_input[i] = input[mirror_idx].real();
        imag_input[i] = -input[mirror_idx].imag();
    }

    // Simple IDFT implementation (very inefficient - for demo only)
    for (uint32_t n = 0; n < fft_size_; ++n) {
        float sum_real = 0.0f;
        float sum_imag = 0.0f;

        for (uint32_t k = 0; k < fft_size_; ++k) {
            float angle = 2.0f * M_PI * k * n / fft_size_;
            sum_real += real_input[k] * std::cos(angle) - imag_input[k] * std::sin(angle);
            sum_imag += real_input[k] * std::sin(angle) + imag_input[k] * std::cos(angle);
        }

        output[n] = sum_real / fft_size_; // Normalize
    }
}

bool FFTProcessor::perform_gpu_fft(const float* input, std::complex<float>* output) {
    // In a real implementation, would use GPU FFT library (cuFFT, clFFT, etc.)
    // For this demo, fall back to CPU
    perform_cpu_fft(input, output);
    return true;
}

bool FFTProcessor::perform_gpu_ifft(const std::complex<float>* input, float* output) {
    // In a real implementation, would use GPU IFFT library
    // For this demo, fall back to CPU
    perform_cpu_ifft(input, output);
    return true;
}

// ParametricEqualizer implementation
ParametricEqualizer::ParametricEqualizer()
    : effect_type_(FrequencyDomainEffectType::EQUALIZER)
    , bypassed_(false)
    , dry_wet_mix_(0.5f)
    , sample_rate_(44100)
    , channels_(2)
    , max_frame_size_(4096)
    , noise_profile_captured_(false) {
}

ParametricEqualizer::~ParametricEqualizer() {
    shutdown();
}

bool ParametricEqualizer::initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    sample_rate_ = sample_rate;
    channels_ = channels;
    max_frame_size_ = max_frame_size;

    Logger::info("Initializing Parametric Equalizer: {} Hz, {} channels", sample_rate, channels);

    // Initialize parameters
    parameters_ = FrequencyDomainParameters();
    parameters_.fft_size = 4096;
    parameters_.overlap_factor = 4;
    parameters_.window_type = FFTWindowType::HANNING;

    // Initialize FFT processor
    fft_processor_ = std::make_unique<FFTProcessor>();
    if (!fft_processor_->initialize(parameters_.fft_size, channels, parameters_.gpu_acceleration_enabled)) {
        Logger::error("Failed to initialize FFT processor");
        return false;
    }

    // Initialize frequency response arrays
    uint32_t num_bins = fft_processor_->get_num_bins();
    frequency_response_.resize(channels, std::vector<float>(num_bins, 1.0f));
    complex_response_.resize(channels, std::vector<std::complex<float>>(num_bins, 1.0f));
    smoothed_response_.resize(channels, std::vector<float>(num_bins, 1.0f));
    response_dirty_.resize(channels, true);

    // Initialize processing buffers
    uint32_t total_samples = max_frame_size * channels;
    input_buffer_.resize(total_samples, 0.0f);
    output_buffer_.resize(total_samples, 0.0f);
    wet_buffer_.resize(total_samples, 0.0f);
    dry_buffer_.resize(total_samples, 0.0f);
    spectrum_.resize(fft_processor_->get_num_bins());

    // Calculate initial frequency response
    calculate_frequency_response();

    // Reset statistics
    statistics_ = FrequencyDomainStatistics();
    statistics_.fft_size = parameters_.fft_size;
    statistics_.overlap_factor = parameters_.overlap_factor;

    // Initialize presets
    presets_["flat"] = parameters_;
    presets_["vocal_boost"] = parameters_;
    presets_["vocal_boost"].eq_bands[1].gain_db = 3.0f;
    presets_["vocal_boost"].eq_bands[1].frequency_hz = 2000.0f;

    Logger::info("Parametric Equalizer initialized with {} bands", parameters_.eq_bands.size());
    return true;
}

void ParametricEqualizer::shutdown() {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    fft_processor_.reset();
    frequency_response_.clear();
    complex_response_.clear();
    smoothed_response_.clear();
    response_dirty_.clear();
    input_buffer_.clear();
    output_buffer_.clear();
    wet_buffer_.clear();
    dry_buffer_.clear();
    spectrum_.clear();
    presets_.clear();

    Logger::info("Parametric Equalizer shutdown");
}

bool ParametricEqualizer::reset() {
    if (fft_processor_) {
        fft_processor_->clear();
    }

    for (auto& response : frequency_response_) {
        std::fill(response.begin(), response.end(), 1.0f);
    }

    for (auto& response : smoothed_response_) {
        std::fill(response.begin(), response.end(), 1.0f);
    }

    std::fill(response_dirty_.begin(), response_dirty_.end(), true);

    Logger::info("Parametric Equalizer reset");
    return true;
}

bool ParametricEqualizer::process(const float* input, float* output, uint32_t frame_count) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (bypassed_ || !input || !output || frame_count == 0) {
        if (input && output && frame_count > 0) {
            std::memcpy(output, input, frame_count * channels_ * sizeof(float));
        }
        return true;
    }

    // Store dry signal
    std::memcpy(dry_buffer_.data(), input, frame_count * channels_ * sizeof(float));
    std::fill(wet_buffer_.begin(), wet_buffer_.begin() + frame_count * channels_, 0.0f);

    // Process each channel
    for (uint32_t ch = 0; ch < channels_; ++ch) {
        // Deinterleave channel data
        std::vector<float> channel_input(frame_count);
        for (uint32_t i = 0; i < frame_count; ++i) {
            channel_input[i] = input[i * channels_ + ch];
        }

        // Forward FFT
        if (!fft_processor_->forward_fft(channel_input.data(), spectrum_.data(), frame_count)) {
            Logger::error("FFT failed for channel {}", ch);
            return false;
        }

        // Apply frequency response
        if (response_dirty_[ch]) {
            smooth_frequency_response(ch);
            response_dirty_[ch] = false;
        }

        apply_frequency_response(spectrum_.data(), spectrum_.data(), ch);

        // Inverse FFT
        std::vector<float> channel_output(parameters_.fft_size);
        if (!fft_processor_->inverse_fft(spectrum_.data(), channel_output.data())) {
            Logger::error("IFFT failed for channel {}", ch);
            return false;
        }

        // Interleave back to output buffer
        for (uint32_t i = 0; i < frame_count; ++i) {
            wet_buffer_[i * channels_ + ch] = channel_output[i];
        }
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
    statistics_.fft_operations += channels_;
    statistics_.ifft_operations += channels_;

    double processing_time_us = duration.count();
    statistics_.avg_processing_time_us =
        ((statistics_.avg_processing_time_us * (statistics_.total_process_calls - 1)) + processing_time_us) /
        statistics_.total_process_calls;
    statistics_.max_processing_time_us = std::max(statistics_.max_processing_time_us, processing_time_us);
    statistics_.min_processing_time_us = std::min(statistics_.min_processing_time_us, processing_time_us);
    statistics_.is_active = true;

    return true;
}

bool ParametricEqualizer::process_interleaved(const float* input, float* output, uint32_t frame_count) {
    return process(input, output, frame_count);
}

bool ParametricEqualizer::set_parameters(const FrequencyDomainParameters& params) {
    parameters_ = params;

    // Update FFT processor if parameters changed
    if (fft_processor_) {
        if (params.fft_size != fft_processor_->get_fft_size()) {
            fft_processor_->set_fft_size(params.fft_size);
        }
        fft_processor_->set_window_type(params.window_type);
        fft_processor_->set_overlap_factor(params.overlap_factor);
    }

    // Recalculate frequency response
    calculate_frequency_response();

    Logger::info("Updated parametric equalizer parameters");
    return true;
}

FrequencyDomainParameters ParametricEqualizer::get_parameters() const {
    return parameters_;
}

bool ParametricEqualizer::set_parameter(const std::string& name, float value) {
    bool response_changed = false;

    if (name == "fft_size") {
        uint32_t new_size = static_cast<uint32_t>(value);
        if (frequency_domain_utils::is_power_of_two(new_size)) {
            parameters_.fft_size = new_size;
            response_changed = true;
        }
    } else if (name == "wet_mix_percent") {
        dry_wet_mix_ = value / 100.0f;
    } else if (name.substr(0, 4) == "band") {
        // Parse band parameter like "band0_gain_db", "band1_frequency_hz", etc.
        size_t band_start = 4;
        size_t underscore_pos = name.find('_', band_start);
        if (underscore_pos != std::string::npos) {
            uint32_t band_index = std::stoul(name.substr(band_start, underscore_pos - band_start));
            std::string param_name = name.substr(underscore_pos + 1);

            if (band_index < parameters_.eq_bands.size()) {
                auto& band = parameters_.eq_bands[band_index];

                if (param_name == "gain_db") {
                    band.gain_db = value;
                    response_changed = true;
                } else if (param_name == "frequency_hz") {
                    band.frequency_hz = value;
                    response_changed = true;
                } else if (param_name == "q_factor") {
                    band.q_factor = value;
                    response_changed = true;
                } else if (param_name == "enabled") {
                    band.enabled = (value != 0.0f);
                    response_changed = true;
                }
            }
        }
    }

    if (response_changed) {
        std::fill(response_dirty_.begin(), response_dirty_.end(), true);
    }

    return true;
}

float ParametricEqualizer::get_parameter(const std::string& name) const {
    if (name == "fft_size") {
        return static_cast<float>(parameters_.fft_size);
    } else if (name == "wet_mix_percent") {
        return dry_wet_mix_ * 100.0f;
    } else if (name.substr(0, 4) == "band") {
        // Parse band parameter
        size_t band_start = 4;
        size_t underscore_pos = name.find('_', band_start);
        if (underscore_pos != std::string::npos) {
            uint32_t band_index = std::stoul(name.substr(band_start, underscore_pos - band_start));
            std::string param_name = name.substr(underscore_pos + 1);

            if (band_index < parameters_.eq_bands.size()) {
                const auto& band = parameters_.eq_bands[band_index];

                if (param_name == "gain_db") {
                    return band.gain_db;
                } else if (param_name == "frequency_hz") {
                    return band.frequency_hz;
                } else if (param_name == "q_factor") {
                    return band.q_factor;
                } else if (param_name == "enabled") {
                    return band.enabled ? 1.0f : 0.0f;
                }
            }
        }
    }

    return 0.0f;
}

bool ParametricEqualizer::set_bypass(bool bypass) {
    bypassed_ = bypass;
    Logger::info("Parametric equalizer bypass {}", bypass ? "enabled" : "disabled");
    return true;
}

bool ParametricEqualizer::is_bypassed() const {
    return bypassed_;
}

bool ParametricEqualizer::set_dry_wet_mix(float mix) {
    dry_wet_mix_ = std::clamp(mix, 0.0f, 1.0f);
    return true;
}

bool ParametricEqualizer::save_preset(const std::string& name) {
    presets_[name] = parameters_;
    Logger::info("Saved preset: {}", name);
    return true;
}

bool ParametricEqualizer::load_preset(const std::string& name) {
    auto it = presets_.find(name);
    if (it != presets_.end()) {
        parameters_ = it->second;
        calculate_frequency_response();
        Logger::info("Loaded preset: {}", name);
        return true;
    }
    Logger::warn("Preset not found: {}", name);
    return false;
}

std::vector<std::string> ParametricEqualizer::get_available_presets() const {
    std::vector<std::string> preset_names;
    for (const auto& preset : presets_) {
        preset_names.push_back(preset.first);
    }
    return preset_names;
}

FrequencyDomainEffectType ParametricEqualizer::get_type() const {
    return effect_type_;
}

std::string ParametricEqualizer::get_name() const {
    return "Parametric Equalizer";
}

std::string ParametricEqualizer::get_version() const {
    return "1.0.0";
}

std::string ParametricEqualizer::get_description() const {
    return "Professional parametric equalizer with FFT-based processing";
}

FrequencyDomainStatistics ParametricEqualizer::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return statistics_;
}

void ParametricEqualizer::reset_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    statistics_ = FrequencyDomainStatistics();
    statistics_.last_reset_time = std::chrono::steady_clock::now();
}

bool ParametricEqualizer::supports_real_time_parameter_changes() const {
    return true;
}

bool ParametricEqualizer::supports_gpu_acceleration() const {
    return true;
}

uint32_t ParametricEqualizer::get_fft_size() const {
    return parameters_.fft_size;
}

double ParametricEqualizer::get_expected_latency_ms() const {
    return (static_cast<double>(parameters_.fft_size) / parameters_.overlap_factor) / sample_rate_ * 1000.0;
}

bool ParametricEqualizer::is_linear_phase() const {
    return false; // This is a minimum-phase implementation
}

void ParametricEqualizer::add_band(float frequency_hz, float gain_db, float q_factor, EQBandType type) {
    FrequencyDomainParameters::EQBand band;
    band.frequency_hz = frequency_hz;
    band.gain_db = gain_db;
    band.q_factor = q_factor;
    band.type = type;
    band.enabled = true;

    parameters_.eq_bands.push_back(band);
    calculate_frequency_response();
}

void ParametricEqualizer::remove_band(uint32_t band_index) {
    if (band_index < parameters_.eq_bands.size()) {
        parameters_.eq_bands.erase(parameters_.eq_bands.begin() + band_index);
        calculate_frequency_response();
    }
}

void ParametricEqualizer::calculate_frequency_response() {
    if (!fft_processor_) {
        return;
    }

    uint32_t num_bins = fft_processor_->get_num_bins();

    for (uint32_t ch = 0; ch < channels_; ++ch) {
        complex_response_[ch].resize(num_bins, std::complex<float>(1.0f, 0.0f));
        frequency_response_[ch].resize(num_bins, 1.0f);

        // Start with flat response
        for (uint32_t i = 0; i < num_bins; ++i) {
            complex_response_[ch][i] = std::complex<float>(1.0f, 0.0f);
        }

        // Apply each band's response
        for (const auto& band : parameters_.eq_bands) {
            if (!band.enabled) {
                continue;
            }

            std::vector<std::complex<float>> band_response(num_bins);
            calculate_band_response(const_cast<FrequencyDomainParameters::EQBand&>(band), band_response);

            // Multiply responses
            for (uint32_t i = 0; i < num_bins; ++i) {
                complex_response_[ch][i] *= band_response[i];
            }
        }

        // Convert to magnitude response
        for (uint32_t i = 0; i < num_bins; ++i) {
            frequency_response_[ch][i] = std::abs(complex_response_[ch][i]);
        }
    }

    std::fill(response_dirty_.begin(), response_dirty_.end(), true);
}

void ParametricEqualizer::calculate_band_response(FrequencyDomainParameters::EQBand& band,
                                                 std::vector<std::complex<float>>& response) {
    if (!fft_processor_) {
        return;
    }

    uint32_t num_bins = fft_processor_->get_num_bins();

    // Calculate band response based on filter type
    for (uint32_t i = 0; i < num_bins; ++i) {
        float frequency_hz = frequency_domain_utils::bin_to_hz(i, fft_processor_->get_fft_size(), sample_rate_);

        float gain_linear = frequency_domain_utils::db_to_magnitude(band.gain_db);
        std::complex<float> filter_response(1.0f, 0.0f);

        switch (band.type) {
            case EQBandType::PEAK: {
                // Peaking filter response
                float omega = 2.0f * M_PI * frequency_hz / sample_rate_;
                float omega0 = 2.0f * M_PI * band.frequency_hz / sample_rate_;
                float alpha = std::sin(omega0) / (2.0f * band.q_factor);

                float A = std::sqrt(gain_linear);
                float cos_omega0 = std::cos(omega0);

                float b0 = 1.0f + alpha * A;
                float b1 = -2.0f * cos_omega0;
                float b2 = 1.0f - alpha * A;
                float a0 = 1.0f + alpha / A;
                float a1 = -2.0f * cos_omega0;
                float a2 = 1.0f - alpha / A;

                // Evaluate filter response at frequency
                std::complex<float> numerator = b0 + b1 * std::exp(std::complex<float>(0, -omega)) +
                                             b2 * std::exp(std::complex<float>(0, -2 * omega));
                std::complex<float> denominator = a0 + a1 * std::exp(std::complex<float>(0, -omega)) +
                                               a2 * std::exp(std::complex<float>(0, -2 * omega));

                filter_response = numerator / denominator;
                break;
            }

            case EQBandType::LOW_SHELF: {
                // Low shelf filter response
                float omega = 2.0f * M_PI * frequency_hz / sample_rate_;
                float omega0 = 2.0f * M_PI * band.frequency_hz / sample_rate_;
                float alpha = std::sin(omega0) / (2.0f * std::sqrt((gain_linear + 1.0f) / (gain_linear - 1.0f)) * band.q_factor);

                float A = std::sqrt(gain_linear);
                float cos_omega0 = std::cos(omega0);

                float b0 = A * ((A + 1.0f) + (A - 1.0f) * cos_omega0 + 2.0f * std::sqrt(A) * alpha);
                float b1 = -2.0f * A * ((A - 1.0f) + (A + 1.0f) * cos_omega0);
                float b2 = A * ((A + 1.0f) + (A - 1.0f) * cos_omega0 - 2.0f * std::sqrt(A) * alpha);
                float a0 = (A + 1.0f) + (A - 1.0f) * cos_omega0 + 2.0f * std::sqrt(A) * alpha;
                float a1 = -2.0f * ((A - 1.0f) + (A + 1.0f) * cos_omega0);
                float a2 = (A + 1.0f) + (A - 1.0f) * cos_omega0 - 2.0f * std::sqrt(A) * alpha;

                std::complex<float> numerator = b0 + b1 * std::exp(std::complex<float>(0, -omega)) +
                                             b2 * std::exp(std::complex<float>(0, -2 * omega));
                std::complex<float> denominator = a0 + a1 * std::exp(std::complex<float>(0, -omega)) +
                                               a2 * std::exp(std::complex<float>(0, -2 * omega));

                filter_response = numerator / denominator;
                break;
            }

            case EQBandType::HIGH_SHELF: {
                // High shelf filter response
                float omega = 2.0f * M_PI * frequency_hz / sample_rate_;
                float omega0 = 2.0f * M_PI * band.frequency_hz / sample_rate_;
                float alpha = std::sin(omega0) / (2.0f * std::sqrt((gain_linear + 1.0f) / (gain_linear - 1.0f)) * band.q_factor);

                float A = std::sqrt(gain_linear);
                float cos_omega0 = std::cos(omega0);

                float b0 = A * ((A + 1.0f) - (A - 1.0f) * cos_omega0 + 2.0f * std::sqrt(A) * alpha);
                float b1 = 2.0f * A * ((A - 1.0f) - (A + 1.0f) * cos_omega0);
                float b2 = A * ((A + 1.0f) - (A - 1.0f) * cos_omega0 - 2.0f * std::sqrt(A) * alpha);
                float a0 = (A + 1.0f) + (A - 1.0f) * cos_omega0 + 2.0f * std::sqrt(A) * alpha;
                float a1 = -2.0f * ((A - 1.0f) + (A + 1.0f) * cos_omega0);
                float a2 = (A + 1.0f) + (A - 1.0f) * cos_omega0 - 2.0f * std::sqrt(A) * alpha;

                std::complex<float> numerator = b0 + b1 * std::exp(std::complex<float>(0, -omega)) +
                                             b2 * std::exp(std::complex<float>(0, -2 * omega));
                std::complex<float> denominator = a0 + a1 * std::exp(std::complex<float>(0, -omega)) +
                                               a2 * std::exp(std::complex<float>(0, -2 * omega));

                filter_response = numerator / denominator;
                break;
            }

            default:
                // Default to flat response
                filter_response = std::complex<float>(1.0f, 0.0f);
                break;
        }

        response[i] = filter_response;
    }
}

void ParametricEqualizer::smooth_frequency_response(uint32_t channel) {
    if (channel >= frequency_response_.size()) {
        return;
    }

    // Simple exponential smoothing
    const float smoothing_factor = 0.1f;

    for (uint32_t i = 0; i < frequency_response_[channel].size(); ++i) {
        smoothed_response_[channel][i] = smoothed_response_[channel][i] * (1.0f - smoothing_factor) +
                                        frequency_response_[channel][i] * smoothing_factor;
    }

    // Update complex response with smoothed magnitudes
    for (uint32_t i = 0; i < complex_response_[channel].size(); ++i) {
        float current_mag = std::abs(complex_response_[channel][i]);
        float current_phase = std::arg(complex_response_[channel][i]);
        complex_response_[channel][i] = smoothed_response_[channel][i] *
                                       std::exp(std::complex<float>(0, current_phase));
    }
}

void ParametricEqualizer::apply_frequency_response(const std::vector<std::complex<float>>& input_spectrum,
                                                   std::vector<std::complex<float>>& output_spectrum,
                                                   uint32_t channel) {
    if (channel >= complex_response_.size()) {
        return;
    }

    uint32_t num_bins = std::min(input_spectrum.size(), complex_response_[channel].size());

    for (uint32_t i = 0; i < num_bins; ++i) {
        output_spectrum[i] = input_spectrum[i] * complex_response_[channel][i];
    }

    // Preserve hermitian symmetry for real signals
    for (uint32_t i = num_bins; i < input_spectrum.size(); ++i) {
        uint32_t mirror_idx = fft_processor_->get_fft_size() - i;
        if (mirror_idx < complex_response_[channel].size()) {
            output_spectrum[i] = input_spectrum[i] * std::conj(complex_response_[channel][mirror_idx]);
        } else {
            output_spectrum[i] = input_spectrum[i];
        }
    }
}

// Factory implementations
std::unique_ptr<FrequencyDomainEffect> FrequencyDomainEffectsFactory::create_parametric_eq(uint32_t bands) {
    auto eq = std::make_unique<ParametricEqualizer>();
    auto params = FrequencyDomainParameters();

    // Create default bands
    params.eq_bands.clear();
    if (bands >= 1) {
        params.eq_bands.push_back({100.0f, 0.0f, 1.0f, EQBandType::LOW_SHELF});
    }
    if (bands >= 2) {
        params.eq_bands.push_back({1000.0f, 0.0f, 1.0f, EQBandType::PEAK});
    }
    if (bands >= 3) {
        params.eq_bands.push_back({10000.0f, 0.0f, 1.0f, EQBandType::HIGH_SHELF});
    }

    eq->set_parameters(params);
    return std::move(eq);
}

std::unique_ptr<FrequencyDomainEffect> FrequencyDomainEffectsFactory::create_multi_band_compressor(uint32_t bands) {
    auto comp = std::make_unique<MultiBandCompressor>();
    auto params = FrequencyDomainParameters();
    params.num_bands = bands;

    // Set default band frequencies
    params.band_frequencies_hz.clear();
    if (bands >= 2) {
        params.band_frequencies_hz = {200.0f, 1000.0f, 4000.0f, 12000.0f};
    }

    comp->set_parameters(params);
    return std::move(comp);
}

std::unique_ptr<FrequencyDomainEffect> FrequencyDomainEffectsFactory::create_spectral_noise_reduction() {
    return std::make_unique<SpectralNoiseReduction>();
}

std::vector<std::string> FrequencyDomainEffectsFactory::get_available_effect_types() {
    return {
        "equalizer", "filter", "multi_band_comp", "spectral_shaper",
        "noise_reduction", "deesser", "dynamic_eq", "spectral_gate",
        "phase_corrector", "linear_phase_filt", "vocoder", "pitch_corrector",
        "harmonic_exciter", "spectral_enhancer", "transient_shaper",
        "stereo_imager", "mid_side_processor", "spectral_compressor",
        "phase_vocoder", "convolution"
    };
}

// Utility namespace implementations
namespace frequency_domain_utils {

float hz_to_bin(float frequency_hz, uint32_t fft_size, uint32_t sample_rate) {
    return frequency_hz * fft_size / sample_rate;
}

float bin_to_hz(uint32_t bin, uint32_t fft_size, uint32_t sample_rate) {
    return static_cast<float>(bin) * sample_rate / fft_size;
}

bool is_power_of_two(uint32_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

uint32_t next_power_of_two(uint32_t value) {
    if (is_power_of_two(value)) {
        return value;
    }
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return value + 1;
}

float magnitude_to_db(float magnitude) {
    return magnitude > 0.0f ? 20.0f * std::log10(magnitude) : -INFINITY;
}

float db_to_magnitude(float db) {
    return std::pow(10.0f, db / 20.0f);
}

void generate_hanning_window(float* window, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (size - 1)));
    }
}

void generate_hamming_window(float* window, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        window[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (size - 1));
    }
}

void generate_blackman_window(float* window, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        window[i] = 0.42f - 0.5f * std::cos(2.0f * M_PI * i / (size - 1)) +
                   0.08f * std::cos(4.0f * M_PI * i / (size - 1));
    }
}

void generate_kaiser_window(float* window, uint32_t size, float beta) {
    // Simplified Kaiser window (would use Bessel functions in real implementation)
    for (uint32_t i = 0; i < size; ++i) {
        float x = static_cast<float>(i) - 0.5f * (size - 1);
        float arg = beta * std::sqrt(1.0f - (4.0f * x * x) / ((size - 1) * (size - 1)));
        window[i] = std::cosh(arg) / std::cosh(beta);
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

} // namespace frequency_domain_utils

} // namespace vortex::core::dsp