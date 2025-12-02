#include "audio_test_harness.hpp"
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <thread>
#include <future>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#endif

namespace vortex::testing {

AudioTestHarness::AudioTestHarness() : rng_(std::random_device{}()) {
    formatManager_.registerBasicFormats();
}

AudioTestHarness::AudioTestHarness(const TestConfiguration& config)
    : config_(config), rng_(std::random_device{}()) {
    formatManager_.registerBasicFormats();
}

AudioTestHarness::~AudioTestHarness() = default;

std::vector<float> AudioTestHarness::generateSineWave(double frequency, double durationSec, double amplitude) const {
    int numSamples = static_cast<int>(config_.sampleRate * durationSec);
    std::vector<float> signal(numSamples);

    for (int i = 0; i < numSamples; ++i) {
        double time = i / config_.sampleRate;
        signal[i] = static_cast<float>(amplitude * std::sin(2.0 * M_PI * frequency * time));
    }

    return signal;
}

std::vector<float> AudioTestHarness::generateWhiteNoise(double durationSec, double amplitude) const {
    int numSamples = static_cast<int>(config_.sampleRate * durationSec);
    std::vector<float> signal(numSamples);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < numSamples; ++i) {
        signal[i] = amplitude * dist(rng_);
    }

    return signal;
}

std::vector<float> AudioTestHarness::generatePinkNoise(double durationSec, double amplitude) const {
    int numSamples = static_cast<int>(config_.sampleRate * durationSec);
    std::vector<float> signal(numSamples);

    // Simple pink noise generator using multiple octaves of white noise
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> buffers(7, 0.0f);
    std::vector<int> counts(7, 0);

    for (int i = 0; i < numSamples; ++i) {
        float sum = 0.0f;
        for (int octave = 0; octave < 7; ++octave) {
            if (i % (1 << octave) == 0) {
                buffers[octave] = dist(rng_);
                counts[octave] = 0;
            }
            sum += buffers[octave];
            counts[octave]++;
        }
        signal[i] = amplitude * sum / 7.0f;
    }

    return signal;
}

std::vector<float> AudioTestHarness::generateSweep(double startFreq, double endFreq, double durationSec, double amplitude) const {
    int numSamples = static_cast<int>(config_.sampleRate * durationSec);
    std::vector<float> signal(numSamples);

    for (int i = 0; i < numSamples; ++i) {
        double time = i / config_.sampleRate;
        double progress = time / durationSec;
        double frequency = startFreq * std::pow(endFreq / startFreq, progress);
        double phase = 2.0 * M_PI * (startFreq * durationSec / std::log(endFreq / startFreq)) *
                       (std::pow(endFreq / startFreq, progress) - 1.0);
        signal[i] = static_cast<float>(amplitude * std::sin(phase));
    }

    return signal;
}

std::vector<float> AudioTestHarness::generateImpulseTrain(double frequency, double durationSec, double amplitude) const {
    int numSamples = static_cast<int>(config_.sampleRate * durationSec);
    std::vector<float> signal(numSamples, 0.0f);

    int period = static_cast<int>(config_.sampleRate / frequency);
    for (int i = 0; i < numSamples; i += period) {
        signal[i] = static_cast<float>(amplitude);
    }

    return signal;
}

std::vector<float> AudioTestHarness::generateMultiTone(const std::vector<double>& frequencies, double durationSec, double amplitude) const {
    int numSamples = static_cast<int>(config_.sampleRate * durationSec);
    std::vector<float> signal(numSamples, 0.0f);

    for (double freq : frequencies) {
        auto tone = generateSineWave(freq, durationSec, amplitude / frequencies.size());
        for (int i = 0; i < numSamples; ++i) {
            signal[i] += tone[i];
        }
    }

    return signal;
}

AudioTestHarness::AudioQualityMetrics AudioTestHarness::analyzeAudioQuality(
    const std::vector<float>& input, const std::vector<float>& output, double sampleRate) const {

    AudioQualityMetrics metrics;

    if (input.size() != output.size()) {
        metrics.signalToNoiseRatioDb = -999.0;
        return metrics;
    }

    // RMS levels
    double inputRms = std::sqrt(std::inner_product(input.begin(), input.end(), input.begin(), 0.0) / input.size());
    double outputRms = std::sqrt(std::inner_product(output.begin(), output.end(), output.begin(), 0.0) / output.size());

    metrics.peakSignalDb = linearToDb(*std::max_element(output.begin(), output.end()));
    metrics.rmsSignalDb = linearToDb(outputRms);
    metrics.dynamicRangeDb = metrics.peakSignalDb - linearToDb(*std::min_element(output.begin(), output.end()));

    // SNR calculation
    std::vector<float> noise(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        noise[i] = output[i] - input[i];
    }
    metrics.signalToNoiseRatioDb = computeSNR(output, noise);

    // THD calculation (assuming input is a single tone)
    auto inputSpectrum = computeSpectrum(input, sampleRate);
    auto outputSpectrum = computeSpectrum(output, sampleRate);

    // Find fundamental frequency
    auto maxIt = std::max_element(inputSpectrum.begin(), inputSpectrum.end());
    size_t fundamentalIdx = std::distance(inputSpectrum.begin(), maxIt);
    double fundamentalFreq = fundamentalIdx * sampleRate / inputSpectrum.size();

    metrics.totalHarmonicDistortionDb = computeTHD(output, fundamentalFreq, sampleRate);

    return metrics;
}

std::vector<std::complex<double>> AudioTestHarness::computeFFT(const std::vector<float>& audio, double sampleRate) const {
    return fft(audio);
}

std::vector<double> AudioTestHarness::computeSpectrum(const std::vector<float>& audio, double sampleRate) const {
    auto fftResult = fft(audio);
    std::vector<double> magnitude(fftResult.size() / 2);

    for (size_t i = 0; i < magnitude.size(); ++i) {
        magnitude[i] = std::abs(fftResult[i]) / audio.size();
    }

    return magnitude;
}

double AudioTestHarness::computeTHD(const std::vector<float>& audio, double fundamentalFrequency, double sampleRate) const {
    auto spectrum = computeSpectrum(audio, sampleRate);
    size_t fundamentalIdx = static_cast<size_t>(fundamentalFrequency * audio.size() / sampleRate);

    if (fundamentalIdx >= spectrum.size()) return -999.0;

    double fundamentalPower = spectrum[fundamentalIdx];
    double harmonicPower = 0.0;

    // Check 2nd to 10th harmonics
    for (int harmonic = 2; harmonic <= 10; ++harmonic) {
        size_t harmonicIdx = static_cast<size_t>(harmonic * fundamentalFrequency * audio.size() / sampleRate);
        if (harmonicIdx < spectrum.size()) {
            harmonicPower += spectrum[harmonicIdx];
        }
    }

    if (fundamentalPower < 1e-10) return -999.0;

    double thdRatio = std::sqrt(harmonicPower / fundamentalPower);
    return linearToDb(thdRatio);
}

double AudioTestHarness::computeSNR(const std::vector<float>& signal, const std::vector<float>& noise) const {
    double signalPower = std::inner_product(signal.begin(), signal.end(), signal.begin(), 0.0) / signal.size();
    double noisePower = std::inner_product(noise.begin(), noise.end(), noise.begin(), 0.0) / noise.size();

    if (noisePower < 1e-10) return 120.0; // Maximum SNR

    return linearToDb(std::sqrt(signalPower / noisePower));
}

std::vector<double> AudioTestHarness::computeFrequencyResponse(
    const std::vector<float>& input, const std::vector<float>& output, double sampleRate) const {

    auto inputSpectrum = computeSpectrum(input, sampleRate);
    auto outputSpectrum = computeSpectrum(output, sampleRate);

    std::vector<double> response(inputSpectrum.size());
    for (size_t i = 0; i < inputSpectrum.size(); ++i) {
        if (inputSpectrum[i] > 1e-10) {
            response[i] = outputSpectrum[i] / inputSpectrum[i];
        } else {
            response[i] = 0.0;
        }
    }

    return response;
}

AudioTestHarness::TestResult AudioTestHarness::testProcessorLatency(AudioProcessor processor, double targetLatencyMs) const {
    TestResult result;
    result.testName = "Processor Latency Test";

    // Generate test signal
    auto testSignal = generateSineWave(1000.0, 0.1); // 100ms test

    auto start = std::chrono::high_resolution_clock::now();
    auto output = processor(testSignal);
    auto end = std::chrono::high_resolution_clock::now();

    result.duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    result.performance.processingTimeMs = result.duration.count() / 1000.0;

    result.passed = result.performance.processingTimeMs <= targetLatencyMs;
    if (!result.passed) {
        result.errorMessage = "Processing latency exceeds target: " +
                             std::to_string(result.performance.processingTimeMs) + "ms > " +
                             std::to_string(targetLatencyMs) + "ms";
    }

    return result;
}

AudioTestHarness::TestResult AudioTestHarness::testAudioQuality(AudioProcessor processor, double testFrequency) const {
    TestResult result;
    result.testName = "Audio Quality Test";

    // Generate test signal
    auto inputSignal = generateSineWave(testFrequency, 1.0); // 1 second

    auto start = std::chrono::high_resolution_clock::now();
    auto outputSignal = processor(inputSignal);
    auto end = std::chrono::high_resolution_clock::now();

    result.duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Analyze quality
    result.quality = analyzeAudioQuality(inputSignal, outputSignal, config_.sampleRate);
    result.performance.processingTimeMs = result.duration.count() / 1000.0;

    // Check quality metrics
    result.passed = (result.quality.signalToNoiseRatioDb >= config_.maxSignalToNoiseRatioDb) &&
                   (result.quality.totalHarmonicDistortionDb >= config_.maxTotalHarmonicDistortionDb);

    if (!result.passed) {
        std::stringstream ss;
        ss << "Audio quality degradation detected:\n"
           << "  SNR: " << result.quality.signalToNoiseRatioDb << " dB (target: " << config_.maxSignalToNoiseRatioDb << " dB)\n"
           << "  THD: " << result.quality.totalHarmonicDistortionDb << " dB (target: " << config_.maxTotalHarmonicDistortionDb << " dB)";
        result.errorMessage = ss.str();
    }

    return result;
}

AudioTestHarness::TestResult AudioTestHarness::testFrequencyResponse(AudioProcessor processor, const std::vector<double>& testFrequencies) const {
    TestResult result;
    result.testName = "Frequency Response Test";

    std::vector<double> frequencies = testFrequencies.empty() ?
        generateFrequencyVector(config_.minTestFrequency, config_.maxTestFrequency, config_.numTestFrequencies) :
        testFrequencies;

    std::vector<double> deviations;
    auto start = std::chrono::high_resolution_clock::now();

    for (double freq : frequencies) {
        auto input = generateSineWave(freq, 0.1);
        auto output = processor(input);
        auto response = computeFrequencyResponse(input, output, config_.sampleRate);

        // Find response at test frequency
        size_t freqIdx = static_cast<size_t>(freq * input.size() / config_.sampleRate);
        if (freqIdx < response.size()) {
            double responseDb = linearToDb(response[freqIdx]);
            double deviation = std::abs(responseDb);
            deviations.push_back(deviation);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    result.quality.frequencyResponseDeviationDb = *std::max_element(deviations.begin(), deviations.end());
    result.performance.processingTimeMs = result.duration.count() / 1000.0;

    result.passed = result.quality.frequencyResponseDeviationDb <= config_.maxFrequencyResponseDeviationDb;

    if (!result.passed) {
        result.errorMessage = "Frequency response deviation exceeds threshold: " +
                             std::to_string(result.quality.frequencyResponseDeviationDb) + " dB > " +
                             std::to_string(config_.maxFrequencyResponseDeviationDb) + " dB";
    }

    return result;
}

AudioTestHarness::TestResult AudioTestHarness::testMemoryUsage(AudioProcessor processor, double maxMemoryMB) const {
    TestResult result;
    result.testName = "Memory Usage Test";

    double initialMemory = getCurrentMemoryUsageMB();
    double peakMemory = initialMemory;

    {
        ScopedMemoryTracker tracker(initialMemory, peakMemory);

        auto testSignal = generateSineWave(1000.0, 1.0);
        auto output = processor(testSignal);
    }

    result.performance.memoryUsageMB = peakMemory - initialMemory;

    result.passed = result.performance.memoryUsageMB <= maxMemoryMB;

    if (!result.passed) {
        result.errorMessage = "Memory usage exceeds limit: " +
                             std::to_string(result.performance.memoryUsageMB) + " MB > " +
                             std::to_string(maxMemoryMB) + " MB";
    }

    return result;
}

bool AudioTestHarness::validateRealtimeConstraint(std::function<void()> operation, double maxTimeMs) const {
    auto start = std::chrono::high_resolution_clock::now();
    operation();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
    return elapsedMs <= maxTimeMs;
}

bool AudioTestHarness::exportTestData(const std::vector<float>& audio, const std::string& filename, double sampleRate) const {
    juce::WavAudioFormat format;
    std::unique_ptr<juce::AudioFormatWriter> writer;

    juce::File file(filename);
    auto outputStream = std::make_unique<juce::FileOutputStream>(file);

    juce::StringPairArray metadata;
    juce::AudioFormatWriter::ThreadedWriter::WritingMode mode = juce::AudioFormatWriter::ThreadedWriter::WritingMode::atomic;

    writer.reset(format.createWriterFor(
        outputStream.get(),
        sampleRate,
        config_.channels,
        config_.bitDepth,
        metadata,
        0));

    if (!writer) return false;

    outputStream.release(); // Writer takes ownership

    juce::AudioBuffer<float> buffer(config_.channels, audio.size() / config_.channels);

    for (int channel = 0; channel < config_.channels; ++channel) {
        buffer.copyFrom(channel, 0, audio.data() + channel, audio.size() / config_.channels, 1, config_.channels);
    }

    return writer->writeFromAudioBuffer(buffer, 0, buffer.getNumSamples());
}

std::vector<double> AudioTestHarness::generateFrequencyVector(double minFreq, double maxFreq, int numFrequencies, bool logarithmic) const {
    std::vector<double> frequencies;
    frequencies.reserve(numFrequencies);

    if (logarithmic) {
        double logMin = std::log10(minFreq);
        double logMax = std::log10(maxFreq);
        double logStep = (logMax - logMin) / (numFrequencies - 1);

        for (int i = 0; i < numFrequencies; ++i) {
            frequencies.push_back(std::pow(10.0, logMin + i * logStep));
        }
    } else {
        double step = (maxFreq - minFreq) / (numFrequencies - 1);
        for (int i = 0; i < numFrequencies; ++i) {
            frequencies.push_back(minFreq + i * step);
        }
    }

    return frequencies;
}

std::vector<std::complex<double>> AudioTestHarness::fft(const std::vector<float>& input) const {
    size_t n = 1;
    while (n < input.size()) n <<= 1;

    std::vector<std::complex<double>> data(n);
    for (size_t i = 0; i < input.size(); ++i) {
        data[i] = std::complex<double>(input[i], 0.0);
    }

    // Cooley-Tukey FFT implementation
    for (size_t len = 2; len <= n; len <<= 1) {
        double angle = -2.0 * M_PI / len;
        std::complex<double> wlen(std::cos(angle), std::sin(angle));

        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0);
            for (size_t j = 0; j < len / 2; ++j) {
                std::complex<double> u = data[i + j];
                std::complex<double> v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    // Bit reversal
    for (size_t i = 0, j = 0; i < n; ++i) {
        if (i < j) {
            std::swap(data[i], data[j]);
        }
        size_t m = n >> 1;
        while (j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    return data;
}

double AudioTestHarness::getCurrentMemoryUsageMB() const {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    }
#elif defined(__linux__)
    long rss = 0L;
    FILE* fp = fopen("/proc/self/statm", "r");
    if (fp) {
        if (fscanf(fp, "%*s%ld", &rss) != 1) {
            rss = 0;
        }
        fclose(fp);
        return static_cast<double>(rss * sysconf(_SC_PAGESIZE)) / (1024.0 * 1024.0);
    }
#elif defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &infoCount) == KERN_SUCCESS) {
        return static_cast<double>(info.resident_size) / (1024.0 * 1024.0);
    }
#endif
    return 0.0;
}

// Static assertion helpers
void AudioTestHarness::assertAudioQuality(const AudioQualityMetrics& metrics, const TestConfiguration& config, const std::string& testName) {
    ASSERT_GE(metrics.signalToNoiseRatioDb, config.maxSignalToNoiseRatioDb)
        << testName << ": SNR " << metrics.signalToNoiseRatioDb << " < " << config.maxSignalToNoiseRatioDb;

    ASSERT_GE(metrics.totalHarmonicDistortionDb, config.maxTotalHarmonicDistortionDb)
        << testName << ": THD " << metrics.totalHarmonicDistortionDb << " > " << config.maxTotalHarmonicDistortionDb;

    ASSERT_LE(metrics.frequencyResponseDeviationDb, config.maxFrequencyResponseDeviationDb)
        << testName << ": Frequency response deviation " << metrics.frequencyResponseDeviationDb << " > " << config.maxFrequencyResponseDeviationDb;
}

void AudioTestHarness::assertPerformance(const PerformanceMetrics& metrics, const TestConfiguration& config, const std::string& testName) {
    ASSERT_LE(metrics.processingTimeMs, config.maxProcessingTimeMs)
        << testName << ": Processing time " << metrics.processingTimeMs << "ms > " << config.maxProcessingTimeMs << "ms";

    ASSERT_LE(metrics.memoryUsageMB, config.maxMemoryUsageMB)
        << testName << ": Memory usage " << metrics.memoryUsageMB << "MB > " << config.maxMemoryUsageMB << "MB";
}

void AudioTestHarness::assertRealtimeConstraint(double actualTimeMs, double maxTimeMs, const std::string& operationName) {
    ASSERT_LE(actualTimeMs, maxTimeMs)
        << operationName << ": " << actualTimeMs << "ms > " << maxTimeMs << "ms";
}

// ScopedMemoryTracker implementation
double ScopedMemoryTracker::getCurrentMemoryUsageMB() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    }
#elif defined(__linux__)
    long rss = 0L;
    FILE* fp = fopen("/proc/self/statm", "r");
    if (fp) {
        if (fscanf(fp, "%*s%ld", &rss) != 1) {
            rss = 0;
        }
        fclose(fp);
        return static_cast<double>(rss * sysconf(_SC_PAGESIZE)) / (1024.0 * 1024.0);
    }
#elif defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &infoCount) == KERN_SUCCESS) {
        return static_cast<double>(info.resident_size) / (1024.0 * 1024.0);
    }
#endif
    return 0.0;
}

} // namespace vortex::testing