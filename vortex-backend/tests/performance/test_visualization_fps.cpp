#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <memory>
#include <random>
#include <future>
#include "core/dsp/spectrum_analyzer.hpp"
#include "core/dsp/waveform_processor.hpp"
#include "core/dsp/vu_meter.hpp"

using namespace vortex::core::dsp;
using namespace std::chrono;

class VisualizationFPSTest : public ::testing::Test {
protected:
    void SetUp() override {
        sample_rate_ = 44100.0;
        buffer_size_ = 1024;
        channels_ = 2;
        target_fps_ = 60;
        frame_duration_ms_ = 1000.0 / target_fps_;

        // Initialize processors
        spectrum_analyzer_ = std::make_unique<SpectrumAnalyzer>();
        waveform_processor_ = std::make_unique<WaveformProcessor>();
        vu_meter_ = std::make_unique<VUMeter>();

        // Generate test audio data
        generate_test_audio();

        // Performance tracking
        frame_times_.reserve(1000);
        fps_samples_.reserve(1000);
    }

    void TearDown() override {
        spectrum_analyzer_.reset();
        waveform_processor_.reset();
        vu_meter_.reset();
    }

    void generate_test_audio() {
        const size_t total_samples = buffer_size_ * 1000; // 1000 buffers worth
        test_audio_.resize(total_samples * channels_);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> signal_dist(-0.5f, 0.5f);
        std::uniform_real_distribution<float> noise_dist(-0.001f, 0.001f);

        for (size_t i = 0; i < total_samples; ++i) {
            float t = static_cast<float>(i) / sample_rate_;

            // Multi-frequency test signal
            float signal = 0.0f;
            signal += 0.2f * std::sin(2.0f * M_PI * 440.0f * t);    // A4
            signal += 0.15f * std::sin(2.0f * M_PI * 880.0f * t);   // A5
            signal += 0.1f * std::sin(2.0f * M_PI * 1760.0f * t);  // A6
            signal += 0.05f * std::sin(2.0f * M_PI * 3520.0f * t);  // A7
            signal += noise_dist(gen);  // Add noise

            signal_dist(gen); // Random component

            // Interleaved stereo
            test_audio_[i * channels_] = signal;
            test_audio_[i * channels_ + 1] = signal * 0.8f; // Slightly different right channel
        }

        current_audio_pos_ = 0;
    }

    bool get_next_audio_buffer(std::vector<float>& buffer) {
        if (current_audio_pos_ + buffer_size_ > test_audio_.size() / channels_) {
            return false; // No more audio data
        }

        buffer.resize(buffer_size_ * channels_);
        for (size_t i = 0; i < buffer_size_; ++i) {
            size_t source_idx = (current_audio_pos_ + i) * channels_;
            buffer[i * channels_] = test_audio_[source_idx];
            buffer[i * channels_ + 1] = test_audio_[source_idx + 1];
        }

        current_audio_pos_ += buffer_size_;
        return true;
    }

    struct FrameMetrics {
        steady_clock::time_point start_time;
        steady_clock::time_point end_time;
        double processing_time_ms;
        double fps;
        bool frame_dropped;
    };

    void process_visualization_frame(FrameMetrics& metrics) {
        metrics.start_time = steady_clock::now();

        // Get next audio buffer
        std::vector<float> audio_buffer;
        if (!get_next_audio_buffer(audio_buffer)) {
            metrics.frame_dropped = true;
            return;
        }

        try {
            // Process spectrum analysis
            auto spectrum_start = steady_clock::now();
            auto spectrum_data = spectrum_analyzer_->processAudio(audio_buffer.data(), buffer_size_);
            auto spectrum_time = steady_clock::now() - spectrum_start;

            // Process waveform
            auto waveform_start = steady_clock::now();
            auto waveform_data = waveform_processor_->processAudio(audio_buffer.data(), buffer_size_);
            auto waveform_time = steady_clock::now() - waveform_start;

            // Process VU meters
            auto vu_start = steady_clock::now();
            auto vu_levels = vu_meter_->processAudio(audio_buffer.data(), buffer_size_);
            auto vu_time = steady_clock::now() - vu_start;

            metrics.end_time = steady_clock::now();
            metrics.processing_time_ms = duration_cast<microseconds>(metrics.end_time - metrics.start_time).count() / 1000.0;
            metrics.frame_dropped = false;

            // Store processing times for analysis
            spectrum_times_.push_back(duration_cast<microseconds>(spectrum_time).count());
            waveform_times_.push_back(duration_cast<microseconds>(waveform_time).count());
            vu_times_.push_back(duration_cast<microseconds>(vu_time).count());

        } catch (const std::exception& e) {
            metrics.frame_dropped = true;
            metrics.end_time = steady_clock::now();
            metrics.processing_time_ms = duration_cast<microseconds>(metrics.end_time - metrics.start_time).count() / 1000.0;
        }
    }

    double calculate_fps(const std::vector<FrameMetrics>& frames) {
        if (frames.size() < 2) return 0.0;

        auto total_time = frames.back().end_time - frames.front().start_time;
        auto duration_ms = duration_cast<milliseconds>(total_time).count();

        return duration_ms > 0 ? (frames.size() * 1000.0 / duration_ms) : 0.0;
    }

    std::unique_ptr<SpectrumAnalyzer> spectrum_analyzer_;
    std::unique_ptr<WaveformProcessor> waveform_processor_;
    std::unique_ptr<VUMeter> vu_meter_;

    double sample_rate_;
    size_t buffer_size_;
    int channels_;
    int target_fps_;
    double frame_duration_ms_;

    std::vector<float> test_audio_;
    size_t current_audio_pos_;

    std::vector<FrameMetrics> frame_times_;
    std::vector<double> fps_samples_;
    std::vector<long> spectrum_times_;
    std::vector<long> waveform_times_;
    std::vector<long> vu_times_;
};

// Test 1: Baseline 60 FPS performance
TEST_F(VisualizationFPSTest, Baseline60FPSPerformance) {
    // Initialize all processors
    ASSERT_TRUE(spectrum_analyzer_->initialize(sample_rate_, 2048, channels_));
    ASSERT_TRUE(waveform_processor_->initialize(sample_rate_, buffer_size_, channels_));
    ASSERT_TRUE(vu_meter_->initialize(sample_rate_, buffer_size_, channels_));

    const int test_duration_seconds = 5;
    const int target_frames = target_fps_ * test_duration_seconds;

    frame_times_.clear();
    frame_times_.reserve(target_frames);

    auto test_start = steady_clock::now();

    for (int frame = 0; frame < target_frames; ++frame) {
        FrameMetrics metrics;
        process_visualization_frame(metrics);
        frame_times_.push_back(metrics);

        // Maintain target FPS timing
        if (frame < target_frames - 1) { // Don't sleep after last frame
            auto processing_time = metrics.end_time - metrics.start_time;
            auto sleep_time = milliseconds(static_cast<long>(frame_duration_ms_)) -
                            duration_cast<milliseconds>(processing_time);

            if (sleep_time.count() > 0) {
                std::this_thread::sleep_for(sleep_time);
            }
        }
    }

    auto test_end = steady_clock::now();
    auto total_test_time = duration_cast<milliseconds>(test_end - test_start).count();

    // Calculate actual FPS
    double actual_fps = frame_times_.size() * 1000.0 / total_test_time;

    EXPECT_NEAR(actual_fps, target_fps_, target_fps_ * 0.05) // Within 5%
        << "Should achieve target 60 FPS within 5% tolerance";

    // Calculate frame drop rate
    int dropped_frames = std::count_if(frame_times_.begin(), frame_times_.end(),
                                       [](const FrameMetrics& m) { return m.frame_dropped; });
    float drop_rate = static_cast<float>(dropped_frames) / frame_times_.size();

    EXPECT_LT(drop_rate, 0.01f) // Less than 1% frame drops
        << "Frame drop rate should be very low";

    // Analyze processing times
    double avg_processing_time = 0.0;
    double max_processing_time = 0.0;
    for (const auto& frame : frame_times_) {
        if (!frame.frame_dropped) {
            avg_processing_time += frame.processing_time_ms;
            max_processing_time = std::max(max_processing_time, frame.processing_time_ms);
        }
    }
    avg_processing_time /= (frame_times_.size() - dropped_frames);

    EXPECT_LT(avg_processing_time, frame_duration_ms_ * 0.8) // Should use less than 80% of frame time
        << "Average processing time should leave room for system overhead";
    EXPECT_LT(max_processing_time, frame_duration_ms_) // Should never exceed frame time
        << "Maximum processing time should not exceed frame duration";
}

// Test 2: GPU acceleration performance
TEST_F(VisualizationFPSTest, GPUAccelerationPerformance) {
    // Test with different processing modes
    ASSERT_TRUE(spectrum_analyzer_->initialize(sample_rate_, 2048, channels_));
    ASSERT_TRUE(waveform_processor_->initialize(sample_rate_, buffer_size_, channels_));
    ASSERT_TRUE(vu_meter_->initialize(sample_rate_, buffer_size_, channels_));

    const int test_frames = 600; // 10 seconds at 60 FPS

    // Test CPU-only mode
    spectrum_analyzer_->setProcessingMode(SpectrumAnalyzer::ProcessingMode::CPU);
    waveform_processor_->setProcessingMode(WaveformProcessor::ProcessingMode::CPU);

    std::vector<FrameMetrics> cpu_frames;
    cpu_frames.reserve(test_frames);

    auto cpu_start = steady_clock::now();

    for (int frame = 0; frame < test_frames; ++frame) {
        FrameMetrics metrics;
        process_visualization_frame(metrics);
        cpu_frames.push_back(metrics);
    }

    auto cpu_end = steady_clock::now();
    auto cpu_total_time = duration_cast<microseconds>(cpu_end - cpu_start).count();

    // Test GPU mode (if available)
    spectrum_analyzer_->setProcessingMode(SpectrumAnalyzer::ProcessingMode::GPU);
    waveform_processor_->setProcessingMode(WaveformProcessor::ProcessingMode::GPU);

    // Reset audio position
    current_audio_pos_ = 0;

    std::vector<FrameMetrics> gpu_frames;
    gpu_frames.reserve(test_frames);

    auto gpu_start = steady_clock::now();

    for (int frame = 0; frame < test_frames; ++frame) {
        FrameMetrics metrics;
        process_visualization_frame(metrics);
        gpu_frames.push_back(metrics);
    }

    auto gpu_end = steady_clock::now();
    auto gpu_total_time = duration_cast<microseconds>(gpu_end - gpu_start).count();

    // Compare performance
    double cpu_fps = cpu_frames.size() * 1000000.0 / cpu_total_time;
    double gpu_fps = gpu_frames.size() * 1000000.0 / gpu_total_time;

    EXPECT_GT(gpu_fps, cpu_fps * 0.95) // GPU should be at least as fast as CPU
        << "GPU acceleration should not degrade performance";

    // If GPU is available, it should provide some speedup
    if (spectrum_analyzer_->isGPUAvailable() && waveform_processor_->isGPUAvailable()) {
        EXPECT_GT(gpu_fps, cpu_fps * 1.1) // 10% speedup expected
            << "GPU should provide performance improvement";
    }
}

// Test 3: Memory usage and allocation patterns
TEST_F(VisualizationFPSTest, MemoryUsageAndAllocation) {
    ASSERT_TRUE(spectrum_analyzer_->initialize(sample_rate_, 2048, channels_));
    ASSERT_TRUE(waveform_processor_->initialize(sample_rate_, buffer_size_, channels_));
    ASSERT_TRUE(vu_meter_->initialize(sample_rate_, buffer_size_, channels_));

    // Measure memory usage before processing
    size_t initial_memory_usage = getCurrentMemoryUsage();

    const int test_frames = 1200; // 20 seconds

    for (int frame = 0; frame < test_frames; ++frame) {
        FrameMetrics metrics;
        process_visualization_frame(metrics);
        frame_times_.push_back(metrics);

        // Check for memory growth every 100 frames
        if (frame % 100 == 99) {
            size_t current_memory = getCurrentMemoryUsage();
            size_t memory_growth = current_memory - initial_memory_usage;

            // Memory growth should be bounded
            EXPECT_LT(memory_growth, 50 * 1024 * 1024) // Less than 50MB growth
                << "Memory usage should not grow excessively during processing";
        }
    }

    // Final memory check
    size_t final_memory_usage = getCurrentMemoryUsage();
    size_t total_memory_growth = final_memory_usage - initial_memory_usage;

    EXPECT_LT(total_memory_growth, 100 * 1024 * 1024) // Less than 100MB total growth
        << "Total memory growth should be reasonable";

    // Memory should stabilize (no leaks)
    // Force garbage collection if applicable and check again
    std::this_thread::sleep_for(milliseconds(100));
    size_t stabilized_memory = getCurrentMemoryUsage();
    EXPECT_LT(stabilized_memory - final_memory_usage, 10 * 1024 * 1024) // Less than 10MB additional growth
        << "Memory usage should stabilize after processing";
}

// Test 4: Multi-threaded performance
TEST_F(VisualizationFPSTest, MultiThreadedPerformance) {
    ASSERT_TRUE(spectrum_analyzer_->initialize(sample_rate_, 2048, channels_));
    ASSERT_TRUE(waveform_processor_->initialize(sample_rate_, buffer_size_, channels_));
    ASSERT_TRUE(vu_meter_->initialize(sample_rate_, buffer_size_, channels_));

    const int num_threads = 4;
    const int frames_per_thread = 300; // 5 seconds per thread

    std::vector<std::future<std::vector<FrameMetrics>>> thread_futures;
    std::vector<std::thread> threads;

    auto worker_function = [&](int thread_id) -> std::vector<FrameMetrics> {
        std::vector<FrameMetrics> frames;
        frames.reserve(frames_per_thread);

        // Each thread gets its own audio position
        size_t thread_audio_pos = thread_id * frames_per_thread * buffer_size_;

        for (int frame = 0; frame < frames_per_thread; ++frame) {
            FrameMetrics metrics;
            metrics.start_time = steady_clock::now();

            // Get audio for this thread
            std::vector<float> audio_buffer;
            if (thread_audio_pos + buffer_size_ < test_audio_.size() / channels_) {
                audio_buffer.resize(buffer_size_ * channels_);
                for (size_t i = 0; i < buffer_size_; ++i) {
                    size_t source_idx = (thread_audio_pos + i) * channels_;
                    audio_buffer[i * channels_] = test_audio_[source_idx];
                    audio_buffer[i * channels_ + 1] = test_audio_[source_idx + 1];
                }
                thread_audio_pos += buffer_size_;
            }

            // Process (simplified for multithreading test)
            try {
                auto spectrum = spectrum_analyzer_->processAudio(audio_buffer.data(), buffer_size_);
                auto waveform = waveform_processor_->processAudio(audio_buffer.data(), buffer_size_);
                auto vu_levels = vu_meter_->processAudio(audio_buffer.data(), buffer_size_);

                metrics.end_time = steady_clock::now();
                metrics.processing_time_ms = duration_cast<microseconds>(metrics.end_time - metrics.start_time).count() / 1000.0;
                metrics.frame_dropped = false;
            } catch (...) {
                metrics.end_time = steady_clock::now();
                metrics.processing_time_ms = duration_cast<microseconds>(metrics.end_time - metrics.start_time).count() / 1000.0;
                metrics.frame_dropped = true;
            }

            frames.push_back(metrics);
        }

        return frames;
    };

    // Start all threads
    auto multi_start = steady_clock::now();

    for (int i = 0; i < num_threads; ++i) {
        std::promise<std::vector<FrameMetrics>> promise;
        thread_futures.push_back(promise.get_future());
        threads.emplace_back([&promise, i, worker_function]() {
            promise.set_value(worker_function(i));
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    auto multi_end = steady_clock::now();

    // Collect results
    std::vector<FrameMetrics> all_frames;
    for (auto& future : thread_futures) {
        auto thread_frames = future.get();
        all_frames.insert(all_frames.end(), thread_frames.begin(), thread_frames.end());
    }

    // Analyze multi-threaded performance
    auto total_time = duration_cast<milliseconds>(multi_end - multi_start).count();
    double multi_threaded_fps = all_frames.size() * 1000.0 / total_time;

    EXPECT_GT(multi_threaded_fps, target_fps_ * 0.8) // Should achieve reasonable FPS even under load
        << "Multi-threaded processing should maintain reasonable performance";

    // Check thread safety - no crashes or corrupted data
    int valid_frames = std::count_if(all_frames.begin(), all_frames.end(),
                                    [](const FrameMetrics& m) { return !m.frame_dropped; });
    EXPECT_GT(valid_frames, all_frames.size() * 0.95) // At least 95% valid frames
        << "Thread safety should be maintained";
}

// Test 5: Stress test with maximum load
TEST_F(VisualizationFPSTest, MaximumLoadStressTest) {
    // Initialize with maximum settings
    ASSERT_TRUE(spectrum_analyzer_->initialize(sample_rate_, 8192, channels_)); // Larger FFT
    ASSERT_TRUE(waveform_processor_->initialize(sample_rate_, 4096, channels_)); // Larger buffer
    ASSERT_TRUE(vu_meter_->initialize(sample_rate_, buffer_size_, channels_));

    const int stress_test_duration_seconds = 30;
    const int target_frames = target_fps_ * stress_test_duration_seconds;

    frame_times_.clear();
    frame_times_.reserve(target_frames);

    auto stress_start = steady_clock::now();

    for (int frame = 0; frame < target_frames; ++frame) {
        FrameMetrics metrics;
        process_visualization_frame(metrics);
        frame_times_.push_back(metrics);

        // No sleeping - run at maximum speed to stress the system
    }

    auto stress_end = steady_clock::now();
    auto stress_duration = duration_cast<milliseconds>(stress_end - stress_start).count();

    // Calculate maximum achievable FPS under stress
    double max_fps = frame_times_.size() * 1000.0 / stress_duration;

    EXPECT_GT(max_fps, target_fps_) // Should be able to exceed target FPS when not limited
        << "System should be capable of exceeding target FPS under no timing constraints";

    // Check for performance degradation over time
    const int measurement_interval = target_fps_ * 5; // Every 5 seconds
    std::vector<double> interval_fps;

    for (size_t i = measurement_interval; i < frame_times_.size(); i += measurement_interval) {
        auto interval_start = frame_times_[i - measurement_interval].start_time;
        auto interval_end = frame_times_[i].end_time;
        auto interval_time = duration_cast<milliseconds>(interval_end - interval_start).count();
        double interval_fps_value = measurement_interval * 1000.0 / interval_time;
        interval_fps.push_back(interval_fps_value);
    }

    // FPS should be relatively stable (less than 10% variation)
    if (interval_fps.size() > 1) {
        double min_fps = *std::min_element(interval_fps.begin(), interval_fps.end());
        double max_fps = *std::max_element(interval_fps.begin(), interval_fps.end());
        double variation = (max_fps - min_fps) / max_fps;

        EXPECT_LT(variation, 0.1) // Less than 10% variation
            << "FPS should be stable over extended period";
    }
}

// Test 6: Quality vs Performance trade-offs
TEST_F(VisualizationFPSTest, QualityVsPerformanceTradeoffs) {
    std::vector<size_t> fft_sizes = {512, 1024, 2048, 4096, 8192};
    std::vector<double> fps_results;

    for (size_t fft_size : fft_sizes) {
        ASSERT_TRUE(spectrum_analyzer_->initialize(sample_rate_, fft_size, channels_));
        ASSERT_TRUE(waveform_processor_->initialize(sample_rate_, buffer_size_, channels_));
        ASSERT_TRUE(vu_meter_->initialize(sample_rate_, buffer_size_, channels_));

        const int test_frames = 300; // 5 seconds
        current_audio_pos_ = 0; // Reset audio position

        frame_times_.clear();
        frame_times_.reserve(test_frames);

        auto test_start = steady_clock::now();

        for (int frame = 0; frame < test_frames; ++frame) {
            FrameMetrics metrics;
            process_visualization_frame(metrics);
            frame_times_.push_back(metrics);

            // Maintain 60 FPS timing
            auto processing_time = metrics.end_time - metrics.start_time;
            auto sleep_time = milliseconds(static_cast<long>(frame_duration_ms_)) -
                            duration_cast<milliseconds>(processing_time);

            if (sleep_time.count() > 0) {
                std::this_thread::sleep_for(sleep_time);
            }
        }

        auto test_end = steady_clock::now();
        auto test_time = duration_cast<milliseconds>(test_end - test_start).count();
        double fps = test_frames * 1000.0 / test_time;
        fps_results.push_back(fps);
    }

    // Performance should degrade gracefully with quality increase
    EXPECT_GT(fps_results[0], fps_results[4]) // Higher FFT size should result in lower FPS
        << "Performance should degrade with increased quality settings";

    // Even with highest quality, should maintain reasonable performance
    EXPECT_GT(fps_results.back(), target_fps_ * 0.7) // At least 70% of target FPS
        << "Even with highest quality, should maintain reasonable performance";

    // Log quality vs performance results
    for (size_t i = 0; i < fft_sizes.size(); ++i) {
        Logger::info("FFT Size: {}, FPS: {:.2f}", fft_sizes[i], fps_results[i]);
    }
}

// Helper function to get current memory usage (platform-specific)
size_t VisualizationFPSTest::getCurrentMemoryUsage() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
#elif defined(__linux__)
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string label, value, unit;
            iss >> label >> value >> unit;
            return std::stoull(value) * 1024; // Convert kB to bytes
        }
    }
#elif defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &infoCount) == KERN_SUCCESS) {
        return info.resident_size;
    }
#endif
    return 0; // Fallback
}