#include <iostream>
#include <memory>
#include <csignal>
#include <thread>
#include <atomic>

#include "vortex_api.hpp"
#include "audio_types.hpp"
#include "network_types.hpp"

#include "core/audio_engine.hpp"
#include "system/logger.hpp"
#include "system/config_manager.hpp"

using namespace vortex;

// Global flag for graceful shutdown
std::atomic<bool> g_shutdown_requested{false};

void signal_handler(int signal) {
    g_shutdown_requested = true;
    Logger::info("Received signal {}, initiating graceful shutdown", signal);
}

int main(int argc, char* argv[]) {
    // Setup signal handlers for graceful shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        // Initialize logger
        Logger::initialize();
        Logger::info("Vortex GPU Audio Backend Starting...");
        Logger::info("Version: 1.0.0");
        Logger::info("Build Date: {}", __DATE__);
        Logger::info("C++ Standard: {}", __cplusplus);

        // Load configuration
        ConfigManager config_manager;
        if (!config_manager.load(argc, argv)) {
            Logger::error("Failed to load configuration");
            return 1;
        }

        const auto& config = config_manager.get_config();
        Logger::info("Configuration loaded successfully");
        Logger::info("Sample Rate: {} Hz", config.audio.sampleRate);
        Logger::info("Bit Depth: {} bits", config.audio.bitDepth);
        Logger::info("Channels: {}", config.audio.channels);
        Logger::info("GPU Acceleration: {}", config.audio.enableGPU ? "Enabled" : "Disabled");

        // Initialize audio engine
        auto audio_engine = std::make_unique<AudioEngine>();
        if (!audio_engine->initialize(config.audio.sampleRate, config.audio.bufferSize)) {
            Logger::error("Failed to initialize audio engine");
            return 1;
        }
        Logger::info("Audio engine initialized successfully");

        // Enable GPU acceleration if requested
        if (config.audio.enableGPU) {
            if (!audio_engine->enableGPUAcceleration(config.gpu.preferredBackends[0])) {
                Logger::error("Failed to enable GPU acceleration");
                return 1;
            }
            Logger::info("GPU acceleration enabled");
        }

        // Main application loop
        Logger::info("Vortex GPU Audio Backend is running...");
        Logger::info("Press Ctrl+C to stop");

        while (!g_shutdown_requested) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // TODO: Handle application logic here
            // - Process audio files
            // - Handle network requests
            // - Update visualizations
            // - Monitor system performance
        }

        // Graceful shutdown
        Logger::info("Shutting down gracefully...");
        audio_engine->shutdown();

        Logger::info("Vortex GPU Audio Backend stopped");
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown fatal error occurred" << std::endl;
        return 1;
    }
}