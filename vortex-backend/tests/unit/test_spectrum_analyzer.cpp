#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <chrono>
#include "core/dsp/spectrum_analyzer.hpp"

using namespace vortex::core::dsp;
using namespace testing;

class SpectrumAnalyzerTest : public ::testing::Test {
protected:
    void SetUp() override {
        sample_rate_ = 44100.0;
        fft_size_ = 2048;
        channels_ = 2;
        overlap_ratio_ = 0.75; // 75% overlap for smooth visualization

        analyzer_ = std::make_unique<SpectrumAnalyzer>();

        // Create test audio buffer
        test_buffer_size_ = fft_size_ * 4; // 4 FFT windows worth of data
        test_audio_.resize(test_buffer_size_ * channels_);

        // Generate test signal: combination of sine waves
        generate_test_signal();
    }

    void TearDown() override {
        analyzer_.reset();
    }

    void generate_test_signal() {
        // Generate a test signal with known frequency components
        // 1kHz sine wave at -20dB
        // 5kHz sine wave at -30dB
        // 10kHz sine wave at -40dB
        // White noise at -50dB

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> noise_dist(-0.001f, 0.001f);

        for (size_t i = 0; i < test_buffer_size_; ++i) {
            float t = static_cast<float>(i) / sample_rate_;

            // 1kHz sine wave (0.1 amplitude = -20dB)
            float signal_1k = 0.1f * std::sin(2.0f * M_PI * 1000.0f * t);

            // 5kHz sine wave (0.032 amplitude = -30dB)
            float signal_5k = 0.032f * std::sin(2.0f * M_PI * 5000.0f * t);

            // 10kHz sine wave (0.01 amplitude = -40dB)
            float signal_10k = 0.01f * std::sin(2.0f * M_PI * 10000.0f * t);

            // White noise
            float noise = noise_dist(gen);

            // Combine signals
            float sample = signal_1k + signal_5k + signal_10k + noise;

            // Store in interleaved stereo format
            test_audio_[i * channels_] = sample;      // Left channel
            test_audio_[i * channels_ + 1] = sample;  // Right channel
        }
    }

    std::unique_ptr<SpectrumAnalyzer> analyzer_;
    double sample_rate_;
    size_t fft_size_;
    int channels_;
    float overlap_ratio_;
    size_t test_buffer_size_;
    std::vector<float> test_audio_;
};

// Test 1: Basic initialization
TEST_F(SpectrumAnalyzerTest, Initialization) {
    EXPECT_TRUE(analyzer_->initialize(sample_rate_, fft_size_, channels_));
    EXPECT_TRUE(analyzer_->isInitialized());

    // Test invalid parameters
    EXPECT_FALSE(analyzer_->initialize(0.0, fft_size_, channels_));
    EXPECT_FALSE(analyzer_->initialize(sample_rate_, 0, channels_));
    EXPECT_FALSE(analyzer_->initialize(sample_rate_, fft_size_, 0));

    // Test invalid FFT size (not power of 2)
    EXPECT_FALSE(analyzer_->initialize(sample_rate_, 1000, channels_));
}

// Test 2: Frequency resolution
TEST_F(SpectrumAnalyzerTest, FrequencyResolution) {
    ASSERT_TRUE(analyzer_->initialize(sample_rate_, fft_size_, channels_));

    double frequency_resolution = analyzer_->getFrequencyResolution();
    double expected_resolution = sample_rate_ / fft_size_;

    EXPECT_NEAR(frequency_resolution, expected_resolution, 0.1)
        << "Frequency resolution should be sample_rate / fft_size";
}

// Test 3: Basic spectrum analysis
TEST_F(SpectrumAnalyzerTest, BasicSpectrumAnalysis) {
    ASSERT_TRUE(analyzer_->initialize(sample_rate_, fft_size_, channels_));

    // Process test audio
    auto spectrum_data = analyzer_->processAudio(test_audio_.data(), test_buffer_size_);

    EXPECT_FALSE(spectrum_data.empty())
        << "Spectrum data should not be empty";

    // Check data structure
    EXPECT_EQ(spectrum_data.size(), channels_)
        << "Should return spectrum data for each channel";

    for (const auto& channel_spectrum : spectrum_data) {
        EXPECT_EQ(channel_spectrum.size(), fft_size_ / 2 + 1)
            << "Spectrum should contain N/2 + 1 frequency bins";

        // Check for non-zero values (our test signal should produce energy)
        bool has_energy = false;
        for (float magnitude : channel_spectrum) {
            if (magnitude > 0.001f) {
                has_energy = true;
                break;
            }
        }
        EXPECT_TRUE(has_energy)
            << "Spectrum should contain energy from test signal";
    }
}

// Test 4: Peak detection for known frequencies
TEST_F(SpectrumAnalyzerTest, PeakDetection) {
    ASSERT_TRUE(analyzer_->initialize(sample_rate_, fft_size_, channels_));

    auto spectrum_data = analyzer_->processAudio(test_audio_.data(), test_buffer_size_);

    // Test the left channel spectrum
    const auto& spectrum = spectrum_data[0];

    // Find peaks around our test frequencies
    double freq_resolution = sample_rate_ / fft_size_;

    // Convert frequencies to bin indices
    size_t bin_1k = static_cast<size_t>(1000.0 / freq_resolution);
    size_t bin_5k = static_cast<size_t>(5000.0 / freq_resolution);
    size_t bin_10k = static_cast<size_t>(10000.0 / freq_resolution);

    // Check for peaks at expected frequency bins
    float magnitude_1k = spectrum[bin_1k];
    float magnitude_5k = spectrum[bin_5k];
    float magnitude_10k = spectrum[bin_10k];

    // 1kHz should be the strongest (-20dB vs -30dB, -40dB)
    EXPECT_GT(magnitude_1k, magnitude_5k)
        << "1kHz peak should be stronger than 5kHz peak";

    EXPECT_GT(magnitude_5k, magnitude_10k)
        << "5kHz peak should be stronger than 10kHz peak";

    // Check that peaks are significantly above noise floor
    float noise_floor = 0.001f; // Approximate noise level
    EXPECT_GT(magnitude_1k, noise_floor * 10)
        << "1kHz peak should be significantly above noise";
    EXPECT_GT(magnitude_5k, noise_floor * 5)
        << "5kHz peak should be above noise";
    EXPECT_GT(magnitude_10k, noise_floor * 2)
        << "10kHz peak should be above noise";
}

// Test 5: Window function effects
TEST_F(SpectrumAnalyzerTest, WindowFunction) {
    ASSERT_TRUE(analyzer_->initialize(sample_rate_, fft_size_, channels_));

    // Test different window functions
    std::vector<SpectrumAnalyzer::WindowType> window_types = {
        SpectrumAnalyzer::WindowType::Hanning,
        SpectrumAnalyzer::WindowType::Hamming,
        SpectrumAnalyzer::WindowType::Blackman,
        SpectrumAnalyzer::WindowType::Rectangular
    };

    for (auto window_type : window_types) {
        analyzer_->setWindowType(window_type);
        EXPECT_EQ(analyzer_->getWindowType(), window_type);

        auto spectrum_data = analyzer_->processAudio(test_audio_.data(), test_buffer_size_);
        EXPECT_FALSE(spectrum_data.empty())
            << "Spectrum data should not be empty for window type " << static_cast<int>(window_type);
    }
}

// Test 6: Overlap processing
TEST_F(SpectrumAnalyzerTest, OverlapProcessing) {
    ASSERT_TRUE(analyzer_->initialize(sample_rate_, fft_size_, channels_));

    // Test different overlap ratios
    std::vector<float> overlap_ratios = {0.0f, 0.5f, 0.75f, 0.875f};

    for (float overlap : overlap_ratios) {
        analyzer_->setOverlapRatio(overlap);
        EXPECT_NEAR(analyzer_->getOverlapRatio(), overlap, 0.001f);

        auto spectrum_data = analyzer_->processAudio(test_audio_.data(), test_buffer_size_);
        EXPECT_FALSE(spectrum_data.empty())
            << "Spectrum data should not be empty for overlap ratio " << overlap;
    }
}

// Test 7: Logarithmic frequency scaling
TEST_F(SpectrumAnalyzerTest, LogarithmicFrequencyScaling) {
    ASSERT_TRUE(analyzer_->initialize(sample_rate_, fft_size_, channels_));

    // Test logarithmic scaling
    analyzer_->setFrequencyScale(SpectrumAnalyzer::FrequencyScale::Logarithmic);
    EXPECT_EQ(analyzer_->getFrequencyScale(), SpectrumAnalyzer::FrequencyScale::Logarithmic);

    auto spectrum_data = analyzer_->processAudio(test_audio_.data(), test_buffer_size_);

    // Logarithmic scaling should produce fewer bins
    const auto& log_spectrum = spectrum_data[0];
    EXPECT_LT(log_spectrum.size(), fft_size_ / 2 + 1)
        << "Logarithmic scaling should produce fewer bins";

    // Test linear scaling
    analyzer_->setFrequencyScale(SpectrumAnalyzer::FrequencyScale::Linear);
    EXPECT_EQ(analyzer_->getFrequencyScale(), SpectrumAnalyzer::FrequencyScale::Linear);

    auto linear_spectrum_data = analyzer_->processAudio(test_audio_.data(), test_buffer_size_);
    EXPECT_EQ(linear_spectrum_data[0].size(), fft_size_ / 2 + 1)
        << "Linear scaling should produce N/2 + 1 bins";
}

// Test 8: Amplitude scaling
TEST_F(SpectrumAnalyzerTest, AmplitudeScaling) {
    ASSERT_TRUE(analyzer_->initialize(sample_rate_, fft_size_, channels_));

    // Test different amplitude scaling modes
    std::vector<SpectrumAnalyzer::AmplitudeScale> amplitude_scales = {
        SpectrumAnalyzer::AmplitudeScale::Linear,
        SpectrumAnalyzer::AmplitudeScale::Decibel,
        SpectrumAnalyzer::AmplitudeScale::SquareRoot
    };

    auto linear_data = analyzer_->processAudio(test_audio_.data(), test_buffer_size_);

    for (auto scale : amplitude_scales) {
        analyzer_->setAmplitudeScale(scale);
        EXPECT_EQ(analyzer_->getAmplitudeScale(), scale);

        auto scaled_data = analyzer_->processAudio(test_audio_.data(), test_buffer_size_);
        EXPECT_FALSE(scaled_data.empty())
            << "Scaled spectrum data should not be empty";

        if (scale == SpectrumAnalyzer::AmplitudeScale::Decibel) {
            // Decibel values should be negative or zero (for signals <= 1.0)
            for (float value : scaled_data[0]) {
                EXPECT_LE(value, 0.0f)
                    << "Decibel values should be <= 0";
            }
        }
    }
}

// Test 9: Performance benchmark
TEST_F(SpectrumAnalyzerTest, PerformanceBenchmark) {
    ASSERT_TRUE(analyzer_->initialize(sample_rate_, fft_size_, channels_));

    const int num_iterations = 1000;
    const size_t buffer_size = fft_size_;

    // Generate test buffer
    std::vector<float> test_buffer(buffer_size * channels_);
    for (size_t i = 0; i < buffer_size * channels_; ++i) {
        test_buffer[i] = std::sin(2.0f * M_PI * 1000.0f * i / sample_rate_);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
        auto spectrum = analyzer_->processAudio(test_buffer.data(), buffer_size);
        EXPECT_FALSE(spectrum.empty());
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    double avg_time_per_analysis = duration.count() / static_cast<double>(num_iterations);

    // Should be able to process audio in under 1ms for real-time performance
    EXPECT_LT(avg_time_per_analysis, 1000.0)
        << "Average analysis time should be under 1ms for real-time performance";

    // Should handle at least 1000 analyses per second
    double analyses_per_second = 1000000.0 / avg_time_per_analysis;
    EXPECT_GT(analyses_per_second, 1000.0)
        << "Should handle at least 1000 analyses per second";
}

// Test 10: Stereo channel processing
TEST_F(SpectrumAnalyzerTest, StereoChannelProcessing) {
    ASSERT_TRUE(analyzer_->initialize(sample_rate_, fft_size_, 2)); // Explicitly stereo

    // Create different signals for left and right channels
    std::vector<float> stereo_buffer(test_buffer_size_ * 2);

    for (size_t i = 0; i < test_buffer_size_; ++i) {
        float t = static_cast<float>(i) / sample_rate_;

        // Left channel: 1kHz
        stereo_buffer[i * 2] = 0.2f * std::sin(2.0f * M_PI * 1000.0f * t);

        // Right channel: 5kHz
        stereo_buffer[i * 2 + 1] = 0.2f * std::sin(2.0f * M_PI * 5000.0f * t);
    }

    auto spectrum_data = analyzer_->processAudio(stereo_buffer.data(), test_buffer_size_);

    EXPECT_EQ(spectrum_data.size(), 2)
        << "Should return spectrum data for both channels";

    // Find peak bins
    double freq_resolution = sample_rate_ / fft_size_;
    size_t bin_1k = static_cast<size_t>(1000.0 / freq_resolution);
    size_t bin_5k = static_cast<size_t>(5000.0 / freq_resolution);

    // Left channel should have peak at 1kHz
    float left_1k = spectrum_data[0][bin_1k];
    float left_5k = spectrum_data[0][bin_5k];
    EXPECT_GT(left_1k, left_5k)
        << "Left channel should have stronger 1kHz component";

    // Right channel should have peak at 5kHz
    float right_1k = spectrum_data[1][bin_1k];
    float right_5k = spectrum_data[1][bin_5k];
    EXPECT_GT(right_5k, right_1k)
        << "Right channel should have stronger 5kHz component";
}

// Test 11: Noise floor measurement
TEST_F(SpectrumAnalyzerTest, NoiseFloorMeasurement) {
    ASSERT_TRUE(analyzer_->initialize(sample_rate_, fft_size_, channels_));

    // Process silent audio to measure noise floor
    std::vector<float> silent_audio(test_buffer_size_ * channels_, 0.0f);
    auto noise_spectrum = analyzer_->processAudio(silent_audio.data(), test_buffer_size_);

    // Noise floor should be very low
    for (const auto& channel_noise : noise_spectrum) {
        for (float noise_level : channel_noise) {
            EXPECT_LT(noise_level, 0.001f)
                << "Noise floor should be very low for silent input";
        }
    }

    // Test with known noise level
    std::vector<float> noisy_audio(test_buffer_size_ * channels_);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> noise_dist(-0.01f, 0.01f); // -40dB noise

    for (size_t i = 0; i < noisy_audio.size(); ++i) {
        noisy_audio[i] = noise_dist(gen);
    }

    auto measured_noise_spectrum = analyzer_->processAudio(noisy_audio.data(), test_buffer_size_);

    // Measured noise should be close to expected level
    float expected_noise_level = 0.01f / std::sqrt(2.0f); // RMS of uniform distribution
    for (const auto& channel_noise : measured_noise_spectrum) {
        float avg_noise = 0.0f;
        for (float noise : channel_noise) {
            avg_noise += noise;
        }
        avg_noise /= channel_noise.size();

        EXPECT_NEAR(avg_noise, expected_noise_level, expected_noise_level * 0.5)
            << "Measured noise level should be close to expected";
    }
}

// Test 12: Edge cases and error handling
TEST_F(SpectrumAnalyzerTest, EdgeCasesAndErrorHandling) {
    ASSERT_TRUE(analyzer_->initialize(sample_rate_, fft_size_, channels_));

    // Test with zero-length buffer
    auto empty_result = analyzer_->processAudio(nullptr, 0);
    EXPECT_TRUE(empty_result.empty())
        << "Should handle zero-length buffer gracefully";

    // Test with single sample
    std::vector<float> single_sample(channels_, 0.5f);
    auto single_result = analyzer_->processAudio(single_sample.data(), 1);
    EXPECT_TRUE(single_result.empty() || single_result[0].size() > 0)
        << "Should handle single sample gracefully";

    // Test with very large buffer (stress test)
    const size_t large_buffer_size = fft_size_ * 1000;
    std::vector<float> large_buffer(large_buffer_size * channels_);
    for (size_t i = 0; i < large_buffer.size(); ++i) {
        large_buffer[i] = std::sin(2.0f * M_PI * 1000.0f * i / sample_rate_);
    }

    auto large_result = analyzer_->processAudio(large_buffer.data(), large_buffer_size);
    EXPECT_FALSE(large_result.empty())
        << "Should handle large buffers";
}