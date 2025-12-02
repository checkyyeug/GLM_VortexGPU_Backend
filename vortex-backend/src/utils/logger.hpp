#pragma once

#include <string>
#include <memory>
#include <fstream>
#include <mutex>
#include <atomic>
#include <sstream>
#include <chrono>
#include <iomanip>

namespace vortex {

/**
 * @brief High-performance logging system for real-time audio processing
 *
 * This logger is optimized for sub-10ms audio processing environments
 * with minimal latency impact and thread-safe operations.
 */
class Logger {
public:
    enum class Level {
        Trace = 0,
        Debug = 1,
        Info = 2,
        Warning = 3,
        Error = 4,
        Critical = 5
    };

    Logger();
    ~Logger();

    // Initialization
    static bool initialize(const std::string& logFile = "vortex_backend.log",
                          Level minLevel = Level::Info,
                          bool enableConsole = true);
    static void shutdown();
    static bool isInitialized();

    // Configuration
    static void setLevel(Level level);
    static Level getLevel();
    static void setConsoleOutput(bool enable);
    static void setFileOutput(bool enable);
    static void setMaxFileSize(size_t maxSize);
    static void setMaxBackupFiles(size_t maxBackups);

    // Logging methods
    template<typename... Args>
    static void trace(const std::string& format, Args&&... args) {
        log(Level::Trace, format, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void debug(const std::string& format, Args&&... args) {
        log(Level::Debug, format, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void info(const std::string& format, Args&&... args) {
        log(Level::Info, format, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void warning(const std::string& format, Args&&... args) {
        log(Level::Warning, format, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void error(const std::string& format, Args&&... args) {
        log(Level::Error, format, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void critical(const std::string& format, Args&&... args) {
        log(Level::Critical, format, std::forward<Args>(args)...);
    }

    // Performance logging
    static void logLatency(const std::string& operation, double latencyMs);
    static void logThroughput(const std::string& operation, double itemsPerSecond);
    static void logMemoryUsage(const std::string& component, size_t memoryBytes);
    static void logGPUUsage(const std::string& operation, float utilization, float temperature = 0.0f);

    // Audio-specific logging
    static void logAudioProcessing(const std::string& operation, uint32_t samplesProcessed,
                                  double processingTimeMs);
    static void logAudioFile(const std::string& filename, const std::string& operation,
                            bool success = true);
    static void logAudioDevice(const std::string& deviceName, const std::string& operation,
                              bool success = true);

    // Real-time performance metrics
    struct PerformanceMetrics {
        double avgLatency = 0.0;
        double maxLatency = 0.0;
        double minLatency = 1000.0;
        uint64_t totalOperations = 0;
        double throughput = 0.0;
        double errorRate = 0.0;
    };

    static void updatePerformanceMetrics(const std::string& operation, double latencyMs, bool success);
    static PerformanceMetrics getPerformanceMetrics(const std::string& operation);
    static void resetPerformanceMetrics();

    // Flush and sync
    static void flush();
    static void sync();

private:
    template<typename... Args>
    static void log(Level level, const std::string& format, Args&&... args) {
        if (!s_instance || level < s_currentLevel.load()) {
            return;
        }

        try {
            // Format the message
            std::stringstream ss;
            ss << formatMessage(format, std::forward<Args>(args)...);

            // Get timestamp
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) % 1000;

            std::lock_guard<std::mutex> lock(s_instance->m_mutex);

            // Format timestamp and level
            ss << " [" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
            ss << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ";
            ss << "[" << levelToString(level) << "] ";

            std::string message = ss.str();

            // Write to console if enabled
            if (s_consoleEnabled.load()) {
                writeToConsole(level, message);
            }

            // Write to file if enabled
            if (s_fileEnabled.load() && s_instance->m_logFile.is_open()) {
                writeToFile(message);
            }

        } catch (const std::exception& e) {
            // Avoid recursive logging errors
            try {
                std::cerr << "Logger error: " << e.what() << std::endl;
            } catch (...) {
                // Last resort - ignore
            }
        }
    }

    template<typename... Args>
    static std::string formatMessage(const std::string& format, Args&&... args) {
        size_t size = snprintf(nullptr, 0, format.c_str(), args...) + 1;
        std::unique_ptr<char[]> buf(new char[size]);
        snprintf(buf.get(), size, format.c_str(), args...);
        return std::string(buf.get(), buf.get() + size - 1);
    }

    static std::string levelToString(Level level);
    static void writeToConsole(Level level, const std::string& message);
    static void writeToFile(const std::string& message);
    static void rotateLogFile();

    // Instance data
    std::mutex m_mutex;
    std::ofstream m_logFile;
    std::string m_logFilePath;
    size_t m_currentFileSize = 0;
    size_t m_maxFileSize = 10 * 1024 * 1024; // 10MB default
    size_t m_maxBackupFiles = 5;

    // Performance metrics
    struct OperationMetrics {
        double totalLatency = 0.0;
        double maxLatency = 0.0;
        double minLatency = 1000.0;
        uint64_t operationCount = 0;
        uint64_t errorCount = 0;
        std::chrono::steady_clock::time_point lastUpdate;
    };

    std::unordered_map<std::string, OperationMetrics> m_metrics;
    mutable std::mutex m_metricsMutex;

    // Static instance and configuration
    static std::unique_ptr<Logger> s_instance;
    static std::atomic<Level> s_currentLevel;
    static std::atomic<bool> s_consoleEnabled;
    static std::atomic<bool> s_fileEnabled;
    static std::atomic<bool> s_initialized;
};

// Convenience macros for performance-critical code
#define LOG_TRACE(...) Logger::trace(__VA_ARGS__)
#define LOG_DEBUG(...) Logger::debug(__VA_ARGS__)
#define LOG_INFO(...) Logger::info(__VA_ARGS__)
#define LOG_WARN(...) Logger::warning(__VA_ARGS__)
#define LOG_ERROR(...) Logger::error(__VA_ARGS__)
#define LOG_CRITICAL(...) Logger::critical(__VA_ARGS__)

// Performance logging macros
#define LOG_LATENCY(op, ms) Logger::logLatency(op, ms)
#define LOG_THROUGHPUT(op, rate) Logger::logThroughput(op, rate)
#define LOG_MEMORY(comp, bytes) Logger::logMemoryUsage(comp, bytes)
#define LOG_GPU(op, util, temp) Logger::logGPUUsage(op, util, temp)

// Audio-specific logging macros
#define LOG_AUDIO_PROCESSING(op, samples, time) Logger::logAudioProcessing(op, samples, time)
#define LOG_AUDIO_FILE(file, op, success) Logger::logAudioFile(file, op, success)
#define LOG_AUDIO_DEVICE(device, op, success) Logger::logAudioDevice(device, op, success)

} // namespace vortex