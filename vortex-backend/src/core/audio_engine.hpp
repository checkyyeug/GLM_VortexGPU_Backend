#pragma once

#include "vortex_api.hpp"
#include "audio_types.hpp"
#include "network_types.hpp"

#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <unordered_map>

namespace vortex {

// Forward declarations
class AudioProcessor;
class GPUProcessor;
class MemoryManager;
class FormatDetector;
class MetadataExtractor;
class ProcessingChainManager;
class OutputManager;
class HardwareMonitor;
class LatencyAnalyzer;
class PerformanceCounter;
class Logger;
class ConfigManager;

/**
 * @brief Core audio processing engine
 *
 * The AudioEngine class is the heart of the Vortex system, managing all audio
 * processing operations, GPU acceleration, file I/O, and real-time audio
 * processing with <10ms latency requirements.
 */
class AudioEngine : public VortexAPI {
public:
    AudioEngine();
    ~AudioEngine() override;

    // VortexAPI interface implementation
    bool initialize(uint32_t sampleRate, uint32_t bufferSize) override;
    void shutdown() override;
    bool isInitialized() const override;

    // GPU Acceleration
    bool enableGPUAcceleration(GPUBackend backend) override;
    bool disableGPUAcceleration() override;
    bool isGPUAccelerationEnabled() const override;
    std::vector<GPUBackend> getAvailableGPUBackends() const override;
    GPUStatus getGPUStatus() const override;

    // Audio File Operations
    std::string uploadAudioFile(const std::string& filePath) override;
    bool removeAudioFile(const std::string& fileId) override;
    AudioFile getAudioFile(const std::string& fileId) const override;
    std::vector<AudioFile> listAudioFiles() const override;
    AudioMetadata extractMetadata(const std::string& fileId) override;

    // Audio Processing
    bool startProcessing(const std::string& fileId) override;
    bool stopProcessing() override;
    bool isProcessing() const override;
    ProcessingStatus getProcessingStatus(const std::string& fileId) const override;
    float getProcessingProgress(const std::string& fileId) const override;

    // Processing Chain Management
    std::string createProcessingChain(const std::string& name) override;
    bool removeProcessingChain(const std::string& chainId) override;
    ProcessingChain getProcessingChain(const std::string& chainId) const override;
    bool setActiveProcessingChain(const std::string& chainId) override;
    std::string addFilter(const std::string& chainId, FilterType type, uint32_t position) override;
    bool removeFilter(const std::string& chainId, const std::string& filterId) override;
    bool setFilterParameter(const std::string& chainId, const std::string& filterId,
                           const std::string& parameter, float value) override;

    // Output Device Management
    std::vector<OutputDevice> discoverOutputDevices() override;
    bool selectOutputDevice(const std::string& deviceId) override;
    OutputDevice getCurrentOutputDevice() const override;
    std::vector<OutputDevice> getAvailableOutputDevices() const override;

    // Real-time Data Access
    RealTimeData getRealTimeData() const override;
    void setRealTimeDataCallback(std::function<void(const RealTimeData&)> callback) override;

    // Session Management
    std::string createSession() override;
    bool closeSession(const std::string& sessionId) override;
    AudioSession getSession(const std::string& sessionId) const override;
    bool loadAudioFileIntoSession(const std::string& sessionId, const std::string& fileId) override;
    bool startPlayback(const std::string& sessionId) override;
    bool pausePlayback(const std::string& sessionId) override;
    bool stopPlayback(const std::string& sessionId) override;
    bool seekPlayback(const std::string& sessionId, double position) override;

    // Configuration
    bool loadConfiguration(const std::string& configPath) override;
    bool saveConfiguration(const std::string& configPath) override;
    SystemConfiguration getConfiguration() const override;
    bool updateConfiguration(const SystemConfiguration& config) override;

    // System Monitoring
    HardwareStatus getHardwareStatus() const override;
    ProcessingMetrics getProcessingMetrics() const override;
    bool isSystemHealthy() const override;

    // Network Services
    bool startNetworkServices() override;
    bool stopNetworkServices() override;
    bool areNetworkServicesRunning() const override;
    uint16_t getHTTPPort() const override;
    uint16_t getWebSocketPort() const override;

    // Advanced audio operations
    bool loadDSDFile(const std::string& filePath, uint32_t dsdRate);
    bool processRealTimeAudio(AudioBuffer& input, AudioBuffer& output);
    bool setRealTimeProcessingEnabled(bool enabled);
    float getCurrentLatency() const;
    bool isRealTimeProcessingEnabled() const;

    // Memory management
    uint64_t getAudioMemoryUsage() const;
    uint64_t getGPUMemoryUsage() const;
    bool optimizeMemoryUsage();
    void clearCache();

    // Advanced GPU operations
    bool switchGPUBackend(GPUBackend newBackend);
    std::vector<std::string> getAvailableGPUDevices() const;
    bool selectGPUDevice(const std::string& deviceId);
    std::string getCurrentGPUDevice() const;

protected:
    // Internal initialization helpers
    bool initializeAudioDevices();
    bool initializeGPUProcessing();
    bool initializeNetworkServices();
    bool initializeRealTimeProcessing();

    // Processing thread management
    void processingThread();
    void realTimeThread();
    void monitoringThread();

    // Audio processing pipeline
    bool processAudioFile(const std::string& fileId);
    void applyProcessingChain(AudioBuffer& buffer, const std::string& chainId);
    void applyDSDProcessing(AudioBuffer& buffer);

    // Utility methods
    std::string generateFileId();
    std::string generateSessionId();
    std::string generateChainId();
    std::string generateFilterId();
    void updateRealTimeData();
    bool validateAudioFile(const std::string& filePath) const;

private:
    // Core components
    std::unique_ptr<AudioProcessor> m_audioProcessor;
    std::unique_ptr<GPUProcessor> m_gpuProcessor;
    std::unique_ptr<MemoryManager> m_memoryManager;
    std::unique_ptr<FormatDetector> m_formatDetector;
    std::unique_ptr<MetadataExtractor> m_metadataExtractor;
    std::unique_ptr<ProcessingChainManager> m_chainManager;
    std::unique_ptr<OutputManager> m_outputManager;
    std::unique_ptr<HardwareMonitor> m_hardwareMonitor;
    std::unique_ptr<LatencyAnalyzer> m_latencyAnalyzer;
    std::unique_ptr<PerformanceCounter> m_performanceCounter;
    std::unique_ptr<Logger> m_logger;
    std::unique_ptr<ConfigManager> m_configManager;

    // Audio file management
    mutable std::mutex m_audioFilesMutex;
    std::unordered_map<std::string, AudioFile> m_audioFiles;

    // Session management
    mutable std::mutex m_sessionsMutex;
    std::unordered_map<std::string, AudioSession> m_sessions;

    // Processing chains
    mutable std::mutex m_chainsMutex;
    std::unordered_map<std::string, ProcessingChain> m_processingChains;
    std::string m_activeChainId;

    // Output devices
    mutable std::mutex m_devicesMutex;
    std::vector<OutputDevice> m_outputDevices;
    std::string m_selectedDeviceId;

    // Real-time data
    mutable std::mutex m_realTimeDataMutex;
    RealTimeData m_realTimeData;
    std::function<void(const RealTimeData&)> m_realTimeDataCallback;

    // Configuration
    mutable std::mutex m_configMutex;
    SystemConfiguration m_configuration;

    // Processing state
    std::atomic<bool> m_initialized{false};
    std::atomic<bool> m_processing{false};
    std::atomic<bool> m_realTimeEnabled{false};
    std::atomic<bool> m_networkServicesRunning{false};

    // Thread management
    std::thread m_processingThread;
    std::thread m_realTimeThread;
    std::thread m_monitoringThread;
    std::atomic<bool> m_shouldShutdown{false};

    // Thread synchronization
    std::mutex m_stateMutex;
    std::condition_variable m_stateCondition;
    std::queue<std::function<void()>> m_processingQueue;
    mutable std::mutex m_queueMutex;

    // Performance tracking
    std::atomic<float> m_currentLatency{0.0f};
    std::atomic<uint64_t> m_samplesProcessed{0};
    std::atomic<uint32_t> m_droppedFrames{0};

    // Memory tracking
    std::atomic<uint64_t> m_audioMemoryUsage{0};
    std::atomic<uint64_t> m_gpuMemoryUsage{0};

    // Constants
    static constexpr uint32_t MAX_AUDIO_FILES = 1000;
    static constexpr uint32_t MAX_SESSIONS = 100;
    static constexpr uint32_t MAX_PROCESSING_CHAINS = 50;
    static constexpr uint32_t PROCESSING_THREAD_INTERVAL_MS = 1;
    static constexpr uint32_t MONITORING_THREAD_INTERVAL_MS = 1000;
    static constexpr uint32_t MAX_QUEUE_SIZE = 1000;
};

} // namespace vortex