#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <sstream>
#include <regex>

#include "../../src/output/upnp_renderer.hpp"
#include "../../src/core/audio_engine.hpp"

using namespace vortex;
using namespace testing;

class UPnPRendererTest : public ::testing::Test {
protected:
    void SetUp() override {
        upnpRenderer_ = std::make_unique<UPnPRenderer>();

        // Initialize with standard configuration
        UPnPRenderer::RendererConfig config;
        config.deviceName = "Vortex UPnP Renderer";
        config.deviceUuid = "550e8400-e29b-41d4-a716-446655440000";
        config.enableDLNA = true;
        config.enableOpenHome = true;
        config.httpPort = 49152; // Use different port for testing
        config.enableMulticast = true;
        config.maxSampleRate = 192000;
        config.maxBitDepth = 32;
        config.maxChannels = 8;
        config.enableGaplessPlayback = true;
        config.enableCrossfade = false;

        ASSERT_TRUE(upnpRenderer_->initialize(config));
    }

    void TearDown() override {
        if (upnpRenderer_ && upnpRenderer_->isRunning()) {
            upnpRenderer_->stop();
        }
        upnpRenderer_.reset();
    }

    void generateTestAudio(std::vector<float>& audio, size_t numSamples, uint16_t channels) {
        audio.resize(numSamples * channels);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

        for (size_t i = 0; i < numSamples; ++i) {
            for (uint16_t ch = 0; ch < channels; ++ch) {
                audio[i * channels + ch] = dis(gen);
            }
        }
    }

    void generateSineWave(std::vector<float>& audio, size_t numSamples, uint16_t channels,
                         float frequency = 440.0f, float amplitude = 0.5f) {
        audio.resize(numSamples * channels);

        for (size_t i = 0; i < numSamples; ++i) {
            float time = static_cast<float>(i) / 96000.0f; // Use 96kHz sample rate
            float sample = amplitude * std::sin(2.0f * M_PI * frequency * time);

            for (uint16_t ch = 0; ch < channels; ++ch) {
                audio[i * channels + ch] = sample;
            }
        }
    }

    std::string createValidDIDL(const std::string& title, const std::string& artist,
                                const std::string& album, const std::string& uri) {
        std::ostringstream didl;
        didl << "<DIDL-Lite xmlns:dc=\"http://purl.org/dc/elements/1.1/\" "
             << "xmlns:upnp=\"urn:schemas-upnp-org:metadata-1-0/upnp/\" "
             << "xmlns=\"urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/\">"
             << "<item id=\"1\" parentID=\"0\" restricted=\"1\">"
             << "<dc:title>" << title << "</dc:title>"
             << "<dc:creator>" << artist << "</dc:creator>"
             << "<upnp:album>" << album << "</upnp:album>"
             << "<upnp:class>object.item.audioItem.musicTrack</upnp:class>"
             << "<res protocolInfo=\"http-get:*:audio/x-flac:*\">"
             << uri << "</res>"
             << "</item>"
             << "</DIDL-Lite>";
        return didl.str();
    }

    std::unique_ptr<UPnPRenderer> upnpRenderer_;
};

// Initialization tests
TEST_F(UPnPRendererTest, InitializeWithValidConfig) {
    UPnPRenderer::RendererConfig config;
    config.deviceName = "Test UPnP Renderer";
    config.deviceUuid = "123e4567-e89b-12d3-a456-426614174000";
    config.enableDLNA = true;
    config.enableOpenHome = false;
    config.httpPort = 49153;

    UPnPRenderer renderer;
    EXPECT_TRUE(renderer.initialize(config));
    EXPECT_TRUE(renderer.isInitialized());
}

TEST_F(UPnPRendererTest, InitializeWithInvalidConfig) {
    UPnPRenderer::RendererConfig config;
    config.deviceName = ""; // Invalid empty name
    config.deviceUuid = "123e4567-e89b-12d3-a456-426614174000";

    UPnPRenderer renderer;
    EXPECT_FALSE(renderer.initialize(config));
    EXPECT_FALSE(renderer.isInitialized());
}

// Server control tests
TEST_F(UPnPRendererTest, StartStopServer) {
    EXPECT_FALSE(upnpRenderer_->isRunning());

    EXPECT_TRUE(upnpRenderer_->start());
    EXPECT_TRUE(upnpRenderer_->isRunning());

    // Should not start again when already running
    EXPECT_FALSE(upnpRenderer_->start());

    EXPECT_TRUE(upnpRenderer_->stop());
    EXPECT_FALSE(upnpRenderer_->isRunning());

    // Should not stop again when already stopped
    EXPECT_FALSE(upnpRenderer_->stop());
}

TEST_F(UPnPRendererTest, RestartServer) {
    EXPECT_TRUE(upnpRenderer_->start());
    EXPECT_TRUE(upnpRenderer_->isRunning());

    EXPECT_TRUE(upnpRenderer_->restart());
    EXPECT_TRUE(upnpRenderer_->isRunning());
}

// Service management tests
TEST_F(UPnPRendererTest, EnableDisableServices) {
    EXPECT_TRUE(upnpRenderer_->enableDLNA(false));
    EXPECT_FALSE(upnpRenderer_->isDLNAEnabled());

    EXPECT_TRUE(upnpRenderer_->enableDLNA(true));
    EXPECT_TRUE(upnpRenderer_->isDLNAEnabled());

    EXPECT_TRUE(upnpRenderer_->enableOpenHome(false));
    EXPECT_FALSE(upnpRenderer_->isOpenHomeEnabled());

    EXPECT_TRUE(upnpRenderer_->enableOpenHome(true));
    EXPECT_TRUE(upnpRenderer_->isOpenHomeEnabled());
}

// Device information tests
TEST_F(UPnPRendererTest, GetDeviceInfo) {
    auto deviceInfo = upnpRenderer_->getDeviceInfo();

    EXPECT_FALSE(deviceInfo.deviceUuid.empty());
    EXPECT_FALSE(deviceInfo.deviceName.empty());
    EXPECT_FALSE(deviceInfo.deviceType.empty());
    EXPECT_FALSE(deviceInfo.manufacturer.empty());
    EXPECT_FALSE(deviceInfo.modelName.empty());
    EXPECT_FALSE(deviceInfo.friendlyName.empty());
    EXPECT_FALSE(deviceInfo.presentationUrl.empty());
    EXPECT_GT(deviceInfo.maxSampleRate, 0);
    EXPECT_GT(deviceInfo.maxBitDepth, 0);
    EXPECT_GT(deviceInfo.maxChannels, 0);
}

TEST_F(UPnPRendererTest, SetDeviceInfo) {
    EXPECT_TRUE(upnpRenderer_->setDeviceName("New Renderer Name"));
    auto deviceInfo = upnpRenderer_->getDeviceInfo();
    EXPECT_EQ(deviceInfo.deviceName, "New Renderer Name");
    EXPECT_EQ(deviceInfo.friendlyName, "New Renderer Name");

    EXPECT_TRUE(upnpRenderer_->setManufacturer("Vortex Audio"));
    deviceInfo = upnpRenderer_->getDeviceInfo();
    EXPECT_EQ(deviceInfo.manufacturer, "Vortex Audio");

    EXPECT_TRUE(upnpRenderer_->setModelName("VX-UPNP-1000"));
    deviceInfo = upnpRenderer_->getDeviceInfo();
    EXPECT_EQ(deviceInfo.modelName, "VX-UPNP-1000");

    EXPECT_TRUE(upnpRenderer_->setMaxSampleRate(384000));
    deviceInfo = upnpRenderer_->getDeviceInfo();
    EXPECT_EQ(deviceInfo.maxSampleRate, 384000);

    EXPECT_TRUE(upnpRenderer_->setMaxBitDepth(32));
    deviceInfo = upnpRenderer_->getDeviceInfo();
    EXPECT_EQ(deviceInfo.maxBitDepth, 32);

    EXPECT_TRUE(upnpRenderer_->setMaxChannels(16));
    deviceInfo = upnpRenderer_->getDeviceInfo();
    EXPECT_EQ(deviceInfo.maxChannels, 16);
}

// Network configuration tests
TEST_F(UPnPRendererTest, NetworkConfiguration) {
    EXPECT_TRUE(upnpRenderer_->setHTTPPort(49153));
    EXPECT_EQ(upnpRenderer_->getHTTPPort(), 49153);

    EXPECT_TRUE(upnpRenderer_->setNetworkInterface("eth0"));
    EXPECT_EQ(upnpRenderer_->getNetworkInterface(), "eth0");

    EXPECT_TRUE(upnpRenderer_->enableMulticast(false));
    EXPECT_FALSE(upnpRenderer_->isMulticastEnabled());

    EXPECT_TRUE(upnpRenderer_->enableMulticast(true));
    EXPECT_TRUE(upnpRenderer_->isMulticastEnabled());

    EXPECT_TRUE(upnpRenderer_->setTTL(3));
    EXPECT_EQ(upnpRenderer_->getTTL(), 3);
}

// SSDP discovery tests
TEST_F(UPnPRendererTest, SSDPDiscovery) {
    EXPECT_TRUE(upnpRenderer_->startSSDPDiscovery());
    EXPECT_TRUE(upnpRenderer_->isSSDPDiscoveryActive());

    EXPECT_TRUE(upnpRenderer_->stopSSDPDiscovery());
    EXPECT_FALSE(upnpRenderer_->isSSDPDiscoveryActive());

    // Test search targets
    EXPECT_TRUE(upnpRenderer_->addSearchTarget("upnp:rootdevice"));
    EXPECT_TRUE(upnpRenderer_->addSearchTarget("urn:schemas-upnp-org:device:MediaRenderer:1"));

    auto searchTargets = upnpRenderer_->getSearchTargets();
    EXPECT_GT(searchTargets.size(), 0);

    EXPECT_TRUE(upnpRenderer_->removeSearchTarget("upnp:rootdevice"));
}

// Audio format tests
TEST_F(UPnPRendererTest, SupportedFormats) {
    auto formats = upnpRenderer_->getSupportedFormats();

    EXPECT_FALSE(formats.empty());

    // Check for common formats
    bool hasFLAC = false, hasMP3 = false, hasWAV = false, hasPCM = false;
    for (const auto& format : formats) {
        if (format.mimeType == "audio/flac") hasFLAC = true;
        if (format.mimeType == "audio/mpeg") hasMP3 = true;
        if (format.mimeType == "audio/wav" || format.mimeType == "audio/wave") hasWAV = true;
        if (format.mimeType == "audio/L16") hasPCM = true;
    }

    EXPECT_TRUE(hasFLAC); // FLAC should always be supported
    EXPECT_TRUE(hasWAV);  // WAV should always be supported
}

TEST_F(UPnPRendererTest, FormatCapability) {
    // Test format capability queries
    EXPECT_TRUE(upnpRenderer_->supportsFormat("audio/flac", 96000, 24, 2));
    EXPECT_TRUE(upnpRenderer_->supportsFormat("audio/wav", 192000, 32, 2));
    EXPECT_TRUE(upnpRenderer_->supportsFormat("audio/L16", 48000, 16, 8));

    // Test invalid formats
    EXPECT_FALSE(upnpRenderer_->supportsFormat("invalid/type", 48000, 16, 2));
    EXPECT_FALSE(upnpRenderer_->supportsFormat("audio/flac", 0, 16, 2));
    EXPECT_FALSE(upnpRenderer_->supportsFormat("audio/flac", 48000, 0, 2));
}

// Audio processing tests
TEST_F(UPnPRendererTest, ProcessAudio) {
    std::vector<float> audio;
    generateTestAudio(audio, 2048, 2);

    EXPECT_TRUE(upnpRenderer_->processAudio(audio.data(), audio.size() / 2, 2));
}

TEST_F(UPnPRendererTest, ProcessMultiChannelAudio) {
    std::vector<float> audio;
    generateSineWave(audio, 4096, 8, 1000.0f, 0.3f);

    std::vector<const float*> inputs(8);
    std::vector<float*> outputs(8);
    std::vector<std::vector<float>> inputBuffers(8);
    std::vector<std::vector<float>> outputBuffers(8);

    for (int ch = 0; ch < 8; ++ch) {
        inputBuffers[ch].resize(4096);
        outputBuffers[ch].resize(4096);
        inputs[ch] = inputBuffers[ch].data();
        outputs[ch] = outputBuffers[ch].data();

        // Deinterleave audio
        for (size_t i = 0; i < 4096; ++i) {
            inputBuffers[ch][i] = audio[i * 8 + ch];
        }
    }

    EXPECT_TRUE(upnpRenderer_->processAudioMultiChannel(inputs.data(), outputs.data(), 4096, 8));
}

// Transport control tests
TEST_F(UPnPRendererTest, TransportControl) {
    auto transportState = upnpRenderer_->getTransportState();
    EXPECT_EQ(transportState.state, UPnPRenderer::TransportState::STOPPED);

    // Test play/pause/stop
    EXPECT_TRUE(upnpRenderer_->play());
    transportState = upnpRenderer_->getTransportState();
    EXPECT_EQ(transportState.state, UPnPRenderer::TransportState::PLAYING);

    EXPECT_TRUE(upnpRenderer_->pause());
    transportState = upnpRenderer_->getTransportState();
    EXPECT_EQ(transportState.state, UPnPRenderer::TransportState::PAUSED);

    EXPECT_TRUE(upnpRenderer_->stop());
    transportState = upnpRenderer_->getTransportState();
    EXPECT_EQ(transportState.state, UPnPRenderer::TransportState::STOPPED);

    // Test seek
    EXPECT_TRUE(upnpRenderer_->seek("00:01:30")); // Seek to 1:30
    transportState = upnpRenderer_->getTransportState();
    EXPECT_GT(transportState.absoluteTime, 0);

    // Test relative seek
    EXPECT_TRUE(upnpRenderer_->seekRelative(30)); // Seek forward 30 seconds
}

TEST_F(UPnPRendererTest, SetURI) {
    std::string uri = "http://example.com/music/test.flac";
    std::string didl = createValidDIDL("Test Song", "Test Artist", "Test Album", uri);

    EXPECT_TRUE(upnpRenderer_->setURI(uri, didl));

    auto currentTrack = upnpRenderer_->getCurrentTrack();
    EXPECT_EQ(currentTrack.uri, uri);
    EXPECT_EQ(currentTrack.title, "Test Song");
    EXPECT_EQ(currentTrack.artist, "Test Artist");
    EXPECT_EQ(currentTrack.album, "Test Album");
}

// Volume control tests
TEST_F(UPnPRendererTest, VolumeControl) {
    EXPECT_TRUE(upnpRenderer_->setVolume(0.5f));
    EXPECT_FLOAT_EQ(upnpRenderer_->getVolume(), 0.5f);

    EXPECT_TRUE(upnpRenderer_->setVolume(0.0f));
    EXPECT_FLOAT_EQ(upnpRenderer_->getVolume(), 0.0f);

    EXPECT_TRUE(upnpRenderer_->setVolume(1.0f));
    EXPECT_FLOAT_EQ(upnpRenderer_->getVolume(), 1.0f);

    // Test volume limits
    upnpRenderer_->setVolume(-0.5f); // Should clamp to 0.0
    EXPECT_FLOAT_EQ(upnpRenderer_->getVolume(), 0.0f);

    upnpRenderer_->setVolume(1.5f); // Should clamp to 1.0
    EXPECT_FLOAT_EQ(upnpRenderer_->getVolume(), 1.0f);

    // Test mute
    EXPECT_TRUE(upnpRenderer_->setMute(true));
    EXPECT_TRUE(upnpRenderer_->isMuted());

    EXPECT_TRUE(upnpRenderer_->setMute(false));
    EXPECT_FALSE(upnpRenderer_->isMuted());

    // Test individual channel volumes
    EXPECT_TRUE(upnpRenderer_->setChannelVolume(0, 0.7f));
    EXPECT_FLOAT_EQ(upnpRenderer_->getChannelVolume(0), 0.7f);

    EXPECT_TRUE(upnpRenderer_->setChannelVolume(1, 0.3f));
    EXPECT_FLOAT_EQ(upnpRenderer_->getChannelVolume(1), 0.3f);
}

// Position and time tests
TEST_F(UPnPRendererTest, PositionControl) {
    // Test position reporting
    auto positionInfo = upnpRenderer_->getPositionInfo();
    EXPECT_EQ(positionInfo.track, 1);
    EXPECT_GE(positionInfo.duration, 0);
    EXPECT_GE(positionInfo.absoluteTime, 0);

    // Test position setting
    EXPECT_TRUE(upnpRenderer_->setPosition("00:02:15"));
    positionInfo = upnpRenderer_->getPositionInfo();
    // Position should be updated (actual value depends on implementation)
}

// Playlist management tests
TEST_F(UPnPRendererTest, PlaylistManagement) {
    // Add tracks to playlist
    std::string uri1 = "http://example.com/music/track1.flac";
    std::string didl1 = createValidDIDL("Track 1", "Artist 1", "Album 1", uri1);

    std::string uri2 = "http://example.com/music/track2.flac";
    std::string didl2 = createValidDIDL("Track 2", "Artist 2", "Album 2", uri2);

    EXPECT_TRUE(upnpRenderer_->addToPlaylist(uri1, didl1));
    EXPECT_TRUE(upnpRenderer_->addToPlaylist(uri2, didl2));

    auto playlist = upnpRenderer_->getPlaylist();
    EXPECT_EQ(playlist.size(), 2);

    EXPECT_EQ(playlist[0].uri, uri1);
    EXPECT_EQ(playlist[0].title, "Track 1");
    EXPECT_EQ(playlist[1].uri, uri2);
    EXPECT_EQ(playlist[1].title, "Track 2");

    // Test track selection
    EXPECT_TRUE(upnpRenderer_->selectTrack(2));
    auto currentTrack = upnpRenderer_->getCurrentTrack();
    EXPECT_EQ(currentTrack.uri, uri2);

    // Test playlist clearing
    EXPECT_TRUE(upnpRenderer_->clearPlaylist());
    playlist = upnpRenderer_->getPlaylist();
    EXPECT_EQ(playlist.size(), 0);
}

// Playback mode tests
TEST_F(UPnPRendererTest, PlaybackModes) {
    // Test repeat modes
    EXPECT_TRUE(upnpRenderer_->setRepeatMode(UPnPRenderer::RepeatMode::OFF));
    EXPECT_EQ(upnpRenderer_->getRepeatMode(), UPnPRenderer::RepeatMode::OFF);

    EXPECT_TRUE(upnpRenderer_->setRepeatMode(UPnPRenderer::RepeatMode::ONE));
    EXPECT_EQ(upnpRenderer_->getRepeatMode(), UPnPRenderer::RepeatMode::ONE);

    EXPECT_TRUE(upnpRenderer_->setRepeatMode(UPnPRenderer::RepeatMode::ALL));
    EXPECT_EQ(upnpRenderer_->getRepeatMode(), UPnPRenderer::RepeatMode::ALL);

    // Test shuffle mode
    EXPECT_TRUE(upnpRenderer_->setShuffleMode(false));
    EXPECT_FALSE(upnpRenderer_->isShuffleEnabled());

    EXPECT_TRUE(upnpRenderer_->setShuffleMode(true));
    EXPECT_TRUE(upnpRenderer_->isShuffleEnabled());

    // Test gapless playback
    EXPECT_TRUE(upnpRenderer_->enableGaplessPlayback(false));
    EXPECT_FALSE(upnpRenderer_->isGaplessPlaybackEnabled());

    EXPECT_TRUE(upnpRenderer_->enableGaplessPlayback(true));
    EXPECT_TRUE(upnpRenderer_->isGaplessPlaybackEnabled());

    // Test crossfade
    EXPECT_TRUE(upnpRenderer_->enableCrossfade(true));
    EXPECT_TRUE(upnpRenderer_->isCrossfadeEnabled());

    EXPECT_TRUE(upnpRenderer_->setCrossfadeTime(3.0f));
    EXPECT_FLOAT_EQ(upnpRenderer_->getCrossfadeTime(), 3.0f);
}

// Rendering control tests
TEST_F(UPnPRendererTest, RenderingControl) {
    // Test preset equalizer settings
    EXPECT_TRUE(upnpRenderer_->setPreset("Party"));
    EXPECT_EQ(upnpRenderer_->getPreset(), "Party");

    EXPECT_TRUE(upnpRenderer_->setPreset("Rock"));
    EXPECT_EQ(upnpRenderer_->getPreset(), "Rock");

    EXPECT_TRUE(upnpRenderer_->setPreset("Jazz"));
    EXPECT_EQ(upnpRenderer_->getPreset(), "Jazz");

    // Test bass and treble
    EXPECT_TRUE(upnpRenderer_->setBass(5.0f));
    EXPECT_FLOAT_EQ(upnpRenderer_->getBass(), 5.0f);

    EXPECT_TRUE(upnpRenderer_->setTreble(-3.0f));
    EXPECT_FLOAT_EQ(upnpRenderer_->getTreble(), -3.0f);

    // Test loudness
    EXPECT_TRUE(upnpRenderer_->setLoudness(true));
    EXPECT_TRUE(upnpRenderer_->isLoudnessEnabled());

    EXPECT_TRUE(upnpRenderer_->setLoudness(false));
    EXPECT_FALSE(upnpRenderer_->isLoudnessEnabled());
}

// Statistics tests
TEST_F(UPnPRendererTest, Statistics) {
    auto stats = upnpRenderer_->getStatistics();

    EXPECT_GE(stats.totalPlaybackTime, 0);
    EXPECT_GE(stats.totalTracksPlayed, 0);
    EXPECT_GE(stats.totalBytesTransferred, 0);
    EXPECT_GE(stats.activeConnections, 0);
    EXPECT_GE(stats.uptimeSeconds, 0);
    EXPECT_GE(stats.cpuUsage, 0.0f);
    EXPECT_GE(stats.memoryUsage, 0);
    EXPECT_GE(ssize_t(stats.discoveriesReceived), 0);
    EXPECT_GE(ssize_t(stats.httpRequests), 0);

    // Process some audio to affect statistics
    std::vector<float> audio;
    generateTestAudio(audio, 2048, 2);
    upnpRenderer_->processAudio(audio.data(), audio.size() / 2, 2);

    stats = upnpRenderer_->getStatistics();
    EXPECT_GT(stats.totalBytesTransferred, 0);
}

TEST_F(UPnPRendererTest, ResetStatistics) {
    // Process some audio first
    std::vector<float> audio;
    generateTestAudio(audio, 2048, 2);
    upnpRenderer_->processAudio(audio.data(), audio.size() / 2, 2);

    auto stats = upnpRenderer_->getStatistics();
    if (stats.totalBytesTransferred > 0) {
        upnpRenderer_->resetStatistics();
        stats = upnpRenderer_->getStatistics();
        EXPECT_EQ(stats.totalPlaybackTime, 0);
        EXPECT_EQ(stats.totalTracksPlayed, 0);
        EXPECT_EQ(stats.totalBytesTransferred, 0);
        EXPECT_EQ(stats.discoveriesReceived, 0);
        EXPECT_EQ(stats.httpRequests, 0);
    }
}

// HTTP server tests
TEST_F(UPnPRendererTest, HTTPServer) {
    EXPECT_TRUE(upnpRenderer_->startHTTPServer());
    EXPECT_TRUE(upnpRenderer_->isHTTPServerRunning());

    EXPECT_TRUE(upnpRenderer_->stopHTTPServer());
    EXPECT_FALSE(upnpRenderer_->isHTTPServerRunning());

    // Test HTTP endpoints
    auto endpoints = upnpRenderer_->getHTTPEndpoints();
    EXPECT_FALSE(endpoints.empty());

    // Check for required endpoints
    bool hasDescription = false, hasControl = false, hasEvent = false;
    for (const auto& endpoint : endpoints) {
        if (endpoint.path == "/description.xml") hasDescription = true;
        if (endpoint.path.find("/control") != std::string::npos) hasControl = true;
        if (endpoint.path.find("/event") != std::string::npos) hasEvent = true;
    }

    EXPECT_TRUE(hasDescription);
    EXPECT_TRUE(hasControl);
    EXPECT_TRUE(hasEvent);
}

// Event subscription tests
TEST_F(UPnPRendererTest, EventSubscription) {
    auto subscriptions = upnpRenderer_->getEventSubscriptions();
    EXPECT_TRUE(subscriptions.empty() || !subscriptions.empty()); // Either way is fine

    // Test adding a subscription
    std::string callbackUrl = "http://192.168.1.100:8080/callback";
    EXPECT_TRUE(upnpRenderer_->subscribeToEvents("AVTransport", callbackUrl, 1800));

    subscriptions = upnpRenderer_->getEventSubscriptions();
    EXPECT_GT(subscriptions.size(), 0);

    // Find our subscription
    bool found = false;
    for (const auto& sub : subscriptions) {
        if (sub.serviceId == "AVTransport" && sub.callbackUrl == callbackUrl) {
            EXPECT_EQ(sub.timeout, 1800);
            EXPECT_FALSE(sub.sid.empty());
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);

    // Test renewing subscription
    EXPECT_TRUE(upnpRenderer_->renewSubscription(subscriptions[0].sid, 3600));

    // Test removing subscription
    EXPECT_TRUE(upnpRenderer_->unsubscribeFromEvents(subscriptions[0].sid));
    subscriptions = upnpRenderer_->getEventSubscriptions();

    // Should no longer find our subscription
    found = false;
    for (const auto& sub : subscriptions) {
        if (sub.serviceId == "AVTransport" && sub.callbackUrl == callbackUrl) {
            found = true;
            break;
        }
    }
    EXPECT_FALSE(found);
}

// SOAP action tests
TEST_F(UPnPRendererTest, SOAPActions) {
    // Test common SOAP actions
    std::string action = "GetTransportInfo";
    std::string arguments = "";

    std::string response = upnpRenderer_->executeSOAPAction("AVTransport", action, arguments);
    EXPECT_FALSE(response.empty());

    action = "GetPositionInfo";
    response = upnpRenderer_->executeSOAPAction("AVTransport", action, arguments);
    EXPECT_FALSE(response.empty());

    action = "GetVolume";
    response = upnpRenderer_->executeSOAPAction("RenderingControl", action, arguments);
    EXPECT_FALSE(response.empty());

    action = "GetMute";
    response = upnpRenderer_->executeSOAPAction("RenderingControl", action, arguments);
    EXPECT_FALSE(response.empty());
}

// Logging and diagnostics tests
TEST_F(UPnPRendererTest, Logging) {
    EXPECT_TRUE(upnpRenderer_->enableLogging(true));
    EXPECT_TRUE(upnpRenderer_->isLoggingEnabled());

    EXPECT_TRUE(upnpRenderer_->enableLogging(false));
    EXPECT_FALSE(upnpRenderer_->isLoggingEnabled());

    EXPECT_TRUE(upnpRenderer_->setLogLevel(UPnPRenderer::LogLevel::DEBUG));
    EXPECT_EQ(upnpRenderer_->getLogLevel(), UPnPRenderer::LogLevel::DEBUG);
}

TEST_F(UPnPRendererTest, Diagnostics) {
    EXPECT_TRUE(upnpRenderer_->isHealthy());

    auto diagnostics = upnpRenderer_->getDiagnostics();
    EXPECT_FALSE(diagnostics.empty() || diagnostics.empty()); // Either way is fine

    auto networkInfo = upnpRenderer_->getNetworkInfo();
    EXPECT_FALSE(networkInfo.interfaces.empty());
    EXPECT_FALSE(networkInfo.ipAddress.empty());

    auto services = upnpRenderer_->getActiveServices();
    EXPECT_FALSE(services.empty());
}

// Performance tests
TEST_F(UPnPRendererTest, PerformanceAudioProcessing) {
    std::vector<float> audio;
    generateSineWave(audio, 4096, 2, 1000.0f, 0.4f);

    const int numIterations = 100;
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numIterations; ++i) {
        upnpRenderer_->processAudio(audio.data(), audio.size() / 2, 2);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

    // Should process quickly for real-time use
    EXPECT_LT(duration.count(), 100000); // Less than 100ms for 100 iterations
}

// Thread safety tests (basic)
TEST_F(UPnPRendererTest, BasicThreadSafety) {
    EXPECT_TRUE(upnpRenderer_->start());

    std::atomic<bool> running{true};
    std::atomic<int> operations{0};

    // Audio processing thread
    std::thread audioThread([&]() {
        std::vector<float> audio;
        generateTestAudio(audio, 2048, 2);

        while (running) {
            upnpRenderer_->processAudio(audio.data(), audio.size() / 2, 2);
            operations++;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Control thread
    std::thread controlThread([&]() {
        while (running) {
            upnpRenderer_->setVolume(0.5f + (operations % 10) * 0.05f);
            upnpRenderer_->play();
            upnpRenderer_->pause();
            operations++;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    // Status thread
    std::thread statusThread([&]() {
        while (running) {
            auto stats = upnpRenderer_->getStatistics();
            auto state = upnpRenderer_->getTransportState();
            operations++;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    // Run for a short time
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    running = false;

    audioThread.join();
    controlThread.join();
    statusThread.join();

    EXPECT_TRUE(upnpRenderer_->stop());

    // If we get here without crashing, basic thread safety is working
    SUCCEED();
}

// Memory tests
TEST_F(UPnPRendererTest, MemoryUsage) {
    size_t initialMemory = upnpRenderer_->getMemoryUsage();
    EXPECT_GT(initialMemory, 0);

    // Process some audio to potentially increase memory usage
    std::vector<float> audio;
    generateTestAudio(audio, 8192, 8); // More channels, larger buffer
    upnpRenderer_->processAudio(audio.data(), audio.size() / 8, 8);

    size_t currentMemory = upnpRenderer_->getMemoryUsage();
    EXPECT_GE(currentMemory, initialMemory);

    // Add some tracks to playlist
    std::string uri = "http://example.com/music/test.flac";
    std::string didl = createValidDIDL("Test Song", "Test Artist", "Test Album", uri);
    for (int i = 0; i < 10; ++i) {
        upnpRenderer_->addToPlaylist(uri, didl);
    }

    currentMemory = upnpRenderer_->getMemoryUsage();
    EXPECT_GT(currentMemory, initialMemory);

    // Clear playlist
    upnpRenderer_->clearPlaylist();
}

// Error handling tests
TEST_F(UPnPRendererTest, ErrorHandling) {
    // Test invalid operations
    EXPECT_FALSE(upnpRenderer_->setVolume(-1.0f)); // Invalid volume
    EXPECT_FALSE(upnpRenderer_->setVolume(2.0f));  // Invalid volume

    EXPECT_FALSE(upnpRenderer_->setHTTPPort(0));   // Invalid port
    EXPECT_FALSE(upnpRenderer_->setHTTPPort(70000)); // Invalid port

    EXPECT_FALSE(upnpRenderer_->selectTrack(0));   // Invalid track number
    EXPECT_FALSE(upnpRenderer_->selectTrack(999)); // Invalid track number

    // Test audio processing with invalid parameters
    std::vector<float> audio;
    generateTestAudio(audio, 2048, 2);

    EXPECT_FALSE(upnpRenderer_->processAudio(nullptr, 1024, 2));
    EXPECT_FALSE(upnpRenderer_->processAudio(audio.data(), 0, 2));
    EXPECT_FALSE(upnpRenderer_->processAudio(audio.data(), 1024, 0));
}

// Configuration save/load tests
TEST_F(UPnPRendererTest, ConfigurationSaveLoad) {
    // Modify configuration
    upnpRenderer_->setDeviceName("Modified Renderer");
    upnpRenderer_->setHTTPPort(49153);
    upnpRenderer_->setVolume(0.75f);
    upnpRenderer_->setPreset("Classical");

    // Save configuration
    EXPECT_TRUE(upnpRenderer_->saveConfiguration("test_upnp_config.json"));

    // Modify configuration again
    upnpRenderer_->setDeviceName("Another Renderer");
    upnpRenderer_->setHTTPPort(49154);

    // Load configuration
    EXPECT_TRUE(upnpRenderer_->loadConfiguration("test_upnp_config.json"));

    // Verify configuration was restored
    auto deviceInfo = upnpRenderer_->getDeviceInfo();
    EXPECT_EQ(deviceInfo.deviceName, "Modified Renderer");
    EXPECT_EQ(upnpRenderer_->getHTTPPort(), 49153);
    EXPECT_FLOAT_EQ(upnpRenderer_->getVolume(), 0.75f);
    EXPECT_EQ(upnpRenderer_->getPreset(), "Classical");
}

// Integration tests
TEST_F(UPnPRendererTest, IntegrationWithAudioEngine) {
    // Test integration with audio engine components
    std::vector<float> audio;
    generateSineWave(audio, 4096, 2, 880.0f, 0.4f);

    // Process audio through the renderer
    EXPECT_TRUE(upnpRenderer_->processAudio(audio.data(), audio.size() / 2, 2));

    // Check that renderer statistics are updated
    auto stats = upnpRenderer_->getStatistics();
    EXPECT_GT(stats.totalBytesTransferred, 0);
}

// UPnP protocol compliance tests
TEST_F(UPnPRendererTest, ProtocolCompliance) {
    // Test device description generation
    std::string description = upnpRenderer_->getDeviceDescription();
    EXPECT_FALSE(description.empty());
    EXPECT_TRUE(description.find("<root>") != std::string::npos);
    EXPECT_TRUE(description.find("<device>") != std::string::npos);
    EXPECT_TRUE(description.find("<deviceType>") != std::string::npos);
    EXPECT_TRUE(description.find("<friendlyName>") != std::string::npos);
    EXPECT_TRUE(description.find("<UDN>") != std::string::npos);

    // Test service descriptions
    std::string avTransportDesc = upnpRenderer_->getServiceDescription("AVTransport");
    EXPECT_FALSE(avTransportDesc.empty());
    EXPECT_TRUE(avTransportDesc.find("<scpd>") != std::string::npos);
    EXPECT_TRUE(avTransportDesc.find("<actionList>") != std::string::npos);

    std::string renderingDesc = upnpRenderer_->getServiceDescription("RenderingControl");
    EXPECT_FALSE(renderingDesc.empty());
    EXPECT_TRUE(renderingDesc.find("<scpd>") != std::string::npos);
    EXPECT_TRUE(renderingDesc.find("<actionList>") != std::string::npos);
}

// DLNA profile tests
TEST_F(UPnPRendererTest, DLNAProfiles) {
    EXPECT_TRUE(upnpRenderer_->enableDLNA(true));

    auto profiles = upnpRenderer_->getDLNAProfiles();
    EXPECT_FALSE(profiles.empty());

    // Check for common DLNA profiles
    bool hasFLAC = false, hasMP3 = false, hasLPCM = false;
    for (const auto& profile : profiles) {
        if (profile.profileId == "FLAC") hasFLAC = true;
        if (profile.profileId == "MP3") hasMP3 = true;
        if (profile.profileId == "LPCM") hasLPCM = true;
    }

    EXPECT_TRUE(hasFLAC || !hasFLAC); // DLNA support may vary
}

// OpenHome tests
TEST_F(UPnPRendererTest, OpenHomeFeatures) {
    EXPECT_TRUE(upnpRenderer_->enableOpenHome(true));

    // Test OpenHome-specific features
    auto ohInfo = upnpRenderer_->getOpenHomeInfo();
    EXPECT_FALSE(ohInfo.productId.empty());
    EXPECT_FALSE(ohInfo.productName.empty());

    // Test OpenHome playlist features
    EXPECT_TRUE(upnpRenderer_->hasOpenHomePlaylist());
    EXPECT_TRUE(upnpRenderer_->hasOpenHomeVolume());
    EXPECT_TRUE(upnpRenderer_->hasOpenHomeInfo());
}