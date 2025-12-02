#include "ogg_decoder.hpp"
#include <cstring>
#include <algorithm>
#include <stdexcept>

#ifdef VORTEX_ENABLE_OGG
#include <ogg/ogg.h>
#include <vorbis/codec.h>
#endif

namespace vortex::core::fileio {

OGGDecoder::OGGDecoder() : initialized_(false) {
    memset(&vorbisInfo_, 0, sizeof(vorbisInfo_));
    memset(&vorbisComment_, 0, sizeof(vorbisComment_));
    memset(&vorbisDspState_, 0, sizeof(vorbisDspState_));
    memset(&vorbisBlock_, 0, sizeof(vorbisBlock_));
}

OGGDecoder::~OGGDecoder() {
    shutdown();
}

bool OGGDecoder::initialize() {
    if (initialized_) {
        return true;
    }

#ifndef VORTEX_ENABLE_OGG
    Logger::error("OGG support not enabled in build");
    return false;
#else

    // Initialize Ogg sync state
    ogg_sync_init(&oggSyncState_);

    initialized_ = true;
    Logger::info("OGG decoder initialized successfully");
    return true;

#endif // VORTEX_ENABLE_OGG
}

void OGGDecoder::shutdown() {
    if (!initialized_) {
        return;
    }

#ifdef VORTEX_ENABLE_OGG
    // Cleanup Vorbis decoder
    if (vorbisBlock_.internal) {
        vorbis_block_clear(&vorbisBlock_);
    }

    if (vorbisDspState_.internal) {
        vorbis_dsp_clear(&vorbisDspState_);
    }

    if (vorbisComment_.user_comments) {
        vorbis_comment_clear(&vorbisComment_);
    }

    if (vorbisInfo_.channels) {
        vorbis_info_clear(&vorbisInfo_);
    }

    // Cleanup Ogg sync state
    ogg_sync_clear(&oggSyncState_);

    memset(&vorbisInfo_, 0, sizeof(vorbisInfo_));
    memset(&vorbisComment_, 0, sizeof(vorbisComment_));
    memset(&vorbisDspState_, 0, sizeof(vorbisDspState_));
    memset(&vorbisBlock_, 0, sizeof(vorbisBlock_));
#endif // VORTEX_ENABLE_OGG

    initialized_ = false;
    Logger::info("OGG decoder shutdown");
}

bool OGGDecoder::canDecode(const std::string& filePath) const {
    if (!initialized_) {
        return false;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Read OGG signature
    uint8_t signature[4];
    file.read(reinterpret_cast<char*>(signature), 4);
    size_t bytesRead = file.gcount();

    file.close();

    if (bytesRead < 4) {
        return false;
    }

    // Check for OGG signature "OggS"
    return (signature[0] == 'O' && signature[1] == 'g' && signature[2] == 'g' && signature[3] == 'S');
}

std::optional<AudioData> OGGDecoder::decode(const std::string& filePath) {
    if (!initialized_) {
        Logger::error("OGG decoder not initialized");
        return std::nullopt;
    }

#ifndef VORTEX_ENABLE_OGG
    Logger::error("OGG support not enabled in build");
    return std::nullopt;
#else

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        Logger::error("Cannot open OGG file: {}", filePath);
        return std::nullopt;
    }

    try {
        Logger::info("Decoding OGG file: {}", filePath);

        // Initialize decoder structures
        vorbis_info_init(&vorbisInfo_);
        vorbis_comment_init(&vorbisComment_);

        // Read file data
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> fileData(fileSize);
        file.read(reinterpret_cast<char*>(fileData.data()), fileSize);

        if (file.gcount() != static_cast<std::streamsize>(fileSize)) {
            Logger::error("Failed to read OGG file data");
            cleanupDecoder();
            return std::nullopt;
        }

        // Setup file reading callback
        FileReadContext context;
        context.data = fileData.data();
        context.size = fileSize;
        context.position = 0;

        // Process OGG/Vorbis headers
        if (!processOGGHeaders(context)) {
            Logger::error("Failed to process OGG headers");
            cleanupDecoder();
            return std::nullopt;
        }

        // Create audio data structure
        AudioData audioData;
        audioData.sampleRate = static_cast<double>(vorbisInfo_.rate);
        audioData.channels = static_cast<uint16_t>(vorbisInfo_.channels);
        audioData.bitDepth = 32;  // Float output
        audioData.format = AudioFormat::OGG;

        // Estimate output buffer size
        uint64_t estimatedSamples = fileData.size();  // Rough estimate
        size_t bufferSize = estimatedSamples * audioData.channels * sizeof(float);
        audioData.data.reserve(bufferSize);
        float* pcmData = nullptr;

        // Decode OGG/Vorbis data
        if (!decodeOGGData(context, audioData)) {
            Logger::error("Failed to decode OGG data");
            cleanupDecoder();
            return std::nullopt;
        }

        cleanupDecoder();

        Logger::info("OGG decoded successfully: {} samples, {} channels, {:.2f} seconds",
                    audioData.data.size() / (sizeof(float) * audioData.channels),
                    audioData.channels,
                    static_cast<double>(audioData.data.size() / (sizeof(float) * audioData.channels)) / audioData.sampleRate);

        return audioData;

    } catch (const std::exception& e) {
        Logger::error("Exception during OGG decoding: {}", e.what());
        cleanupDecoder();
        return std::nullopt;
    }

#endif // VORTEX_ENABLE_OGG
}

bool OGGDecoder::extractMetadata(const std::string& filePath, AudioMetadata& metadata) {
    if (!initialized_) {
        return false;
    }

#ifndef VORTEX_ENABLE_OGG
    Logger::error("OGG support not enabled in build");
    return false;
#else

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    try {
        // Initialize metadata structure
        metadata.format = AudioFormat::OGG;
        metadata.codec = "Vorbis";

        // Read file data
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> fileData(fileSize);
        file.read(reinterpret_cast<char*>(fileData.data()), fileSize);

        // Initialize decoder for metadata extraction
        vorbis_info_init(&vorbisInfo_);
        vorbis_comment_init(&vorbisComment_);

        // Setup file reading context
        FileReadContext context;
        context.data = fileData.data();
        context.size = fileSize;
        context.position = 0;

        // Process headers to get metadata
        if (processOGGHeaders(context)) {
            // Extract technical metadata
            metadata.sampleRate = static_cast<uint32_t>(vorbisInfo_.rate);
            metadata.channels = static_cast<uint16_t>(vorbisInfo_.channels);
            metadata.bitDepth = 32;  // Float output

            // Extract Vorbis comments
            extractVorbisComments(metadata);
        }

        cleanupDecoder();
        return true;

    } catch (const std::exception& e) {
        Logger::error("Exception during OGG metadata extraction: {}", e.what());
        cleanupDecoder();
        return false;
    }

#endif // VORTEX_ENABLE_OGG
}

bool OGGDecoder::processOGGHeaders(FileReadContext& context) {
    ogg_page oggPage;
    ogg_packet oggPacket;
    ogg_stream_state oggStream;
    bool headersComplete = false;
    int headerPackets = 0;

    while (!headersComplete && headerPackets < 3) {
        // Read OGG page
        if (!readOGGPage(context, oggPage)) {
            Logger::error("Failed to read OGG page");
            return false;
        }

        // For the first packet, initialize stream
        if (headerPackets == 0) {
            int serialNo = ogg_page_serialno(&oggPage);
            ogg_stream_init(&oggStream, serialNo);
        }

        // Add page to stream
        if (ogg_stream_pagein(&oggStream, &oggPage) != 0) {
            Logger::error("Failed to add OGG page to stream");
            ogg_stream_clear(&oggStream);
            return false;
        }

        // Extract packets from page
        while (ogg_stream_packetout(&oggStream, &oggPacket) > 0) {
            int result = 0;
            switch (headerPackets) {
                case 0:  // Identification header
                    result = vorbis_synthesis_headerin(&vorbisInfo_, &vorbisComment_, &oggPacket);
                    if (result == 0) {
                        headerPackets++;
                        Logger::debug("Processed OGG identification header");
                    } else {
                        Logger::error("Failed to process OGG identification header: {}", result);
                        ogg_stream_clear(&oggStream);
                        return false;
                    }
                    break;

                case 1:  // Comment header
                    result = vorbis_synthesis_headerin(&vorbisInfo_, &vorbisComment_, &oggPacket);
                    if (result == 0) {
                        headerPackets++;
                        Logger::debug("Processed OGG comment header");
                    } else {
                        Logger::error("Failed to process OGG comment header: {}", result);
                        ogg_stream_clear(&oggStream);
                        return false;
                    }
                    break;

                case 2:  // Setup header
                    result = vorbis_synthesis_headerin(&vorbisInfo_, &vorbisComment_, &oggPacket);
                    if (result == 0) {
                        headerPackets++;
                        headersComplete = true;
                        Logger::debug("Processed OGG setup header");
                    } else {
                        Logger::error("Failed to process OGG setup header: {}", result);
                        ogg_stream_clear(&oggStream);
                        return false;
                    }
                    break;

                default:
                    break;
            }
        }
    }

    if (!headersComplete) {
        Logger::error("Incomplete OGG headers");
        ogg_stream_clear(&oggStream);
        return false;
    }

    // Initialize synthesis
    if (vorbis_synthesis_init(&vorbisDspState_, &vorbisInfo_) != 0) {
        Logger::error("Failed to initialize Vorbis synthesis");
        ogg_stream_clear(&oggStream);
        return false;
    }

    if (vorbis_block_init(&vorbisDspState_, &vorbisBlock_) != 0) {
        Logger::error("Failed to initialize Vorbis block");
        vorbis_synthesis_clear(&vorbisDspState_);
        ogg_stream_clear(&oggStream);
        return false;
    }

    ogg_stream_clear(&oggStream);
    return true;
}

bool OGGDecoder::decodeOGGData(FileReadContext& context, AudioData& audioData) {
    ogg_page oggPage;
    ogg_packet oggPacket;
    ogg_stream_state oggStream;

    // Initialize stream with any serial number from the remaining data
    if (readOGGPage(context, oggPage)) {
        int serialNo = ogg_page_serialno(&oggPage);
        ogg_stream_init(&oggStream, serialNo);

        // Put back the page we read for initialization
        ogg_stream_pagein(&oggStream, &oggPage);
    }

    bool eos = false;
    float** pcm = nullptr;

    while (!eos && context.position < context.size) {
        // Read OGG page
        while (!eos) {
            int result = ogg_stream_packetout(&oggStream, &oggPacket);

            if (result == 0) {
                // Need more data
                if (!readOGGPage(context, oggPage)) {
                    eos = true;
                    break;
                }

                if (ogg_stream_pagein(&oggStream, &oggPage) != 0) {
                    Logger::warn("Failed to add OGG page to stream during decode");
                    eos = true;
                    break;
                }
                continue;
            }

            if (result < 0) {
                Logger::warn("Corrupt OGG packet, continuing");
                continue;
            }

            // Decode packet
            if (vorbis_synthesis(&vorbisBlock_, &oggPacket) == 0) {
                if (vorbis_synthesis_blockin(&vorbisDspState_, &vorbisBlock_) == 0) {
                    int samples;
                    while ((samples = vorbis_synthesis_pcmout(&vorbisDspState_, &pcm)) > 0) {
                        // Convert float PCM to output buffer
                        size_t currentSamples = audioData.data.size() / (sizeof(float) * audioData.channels);
                        size_t newSamples = currentSamples + samples;
                        size_t newSize = newSamples * audioData.channels * sizeof(float);

                        audioData.data.resize(newSize);
                        float* outputPtr = reinterpret_cast<float*>(audioData.data.data()) + currentSamples * audioData.channels;

                        // Interleave channels
                        for (int channel = 0; channel < vorbisInfo_.channels; ++channel) {
                            for (int sample = 0; sample < samples; ++sample) {
                                outputPtr[sample * vorbisInfo_.channels + channel] = pcm[channel][sample];
                            }
                        }

                        vorbis_synthesis_read(&vorbisDspState_, samples);
                    }
                }
            }
        }

        if (ogg_page_eos(&oggPage)) {
            eos = true;
        }
    }

    ogg_stream_clear(&oggStream);
    return true;
}

bool OGGDecoder::readOGGPage(FileReadContext& context, ogg_page& page) {
    if (context.position >= context.size) {
        return false;
    }

    // Look for OGG page header
    while (context.position < context.size - 27) {
        if (context.data[context.position] == 'O' &&
            context.data[context.position + 1] == 'g' &&
            context.data[context.position + 2] == 'g' &&
            context.data[context.position + 3] == 'S') {

            // Found OGG page header
            uint8_t header[27];
            std::memcpy(header, context.data + context.position, 27);

            // Read page size from header
            uint32_t pageSegments = header[26];
            size_t headerSize = 27 + pageSegments;

            if (context.position + headerSize > context.size) {
                return false;  // Incomplete page
            }

            // Calculate total page size
            size_t totalSize = headerSize;
            for (uint32_t i = 0; i < pageSegments; ++i) {
                totalSize += context.data[context.position + 27 + i];
            }

            if (context.position + totalSize > context.size) {
                return false;  // Incomplete page
            }

            // Create ogg_page structure
            char* pageData = const_cast<char*>(reinterpret_cast<const char*>(context.data + context.position));
            ogg_page_init(&page, pageData);

            context.position += totalSize;
            return true;
        }
        context.position++;
    }

    return false;  // No more pages found
}

void OGGDecoder::extractVorbisComments(AudioMetadata& metadata) {
    for (int i = 0; i < vorbisComment_.comments; ++i) {
        std::string comment(vorbisComment_.user_comments[i], vorbisComment_.comment_lengths[i]);

        // Parse Vorbis comment format: "KEY=value"
        size_t separatorPos = comment.find('=');
        if (separatorPos != std::string::npos) {
            std::string key = comment.substr(0, separatorPos);
            std::string value = comment.substr(separatorPos + 1);

            // Normalize key to uppercase
            std::transform(key.begin(), key.end(), key.begin(), ::toupper);

            // Map common Vorbis comment keys
            if (key == "TITLE") {
                metadata.title = value;
            } else if (key == "ARTIST") {
                metadata.artist = value;
            } else if (key == "ALBUM") {
                metadata.album = value;
            } else if (key == "DATE") {
                try {
                    metadata.year = static_cast<uint16_t>(std::stoi(value.substr(0, 4)));
                } catch (...) {
                    // Invalid year format
                }
            } else if (key == "GENRE") {
                metadata.genre = value;
            } else if (key == "TRACKNUMBER") {
                try {
                    metadata.track = static_cast<uint16_t>(std::stoi(value));
                } catch (...) {
                    // Invalid track number
                }
            }
        }
    }
}

void OGGDecoder::cleanupDecoder() {
#ifdef VORTEX_ENABLE_OGG
    if (vorbisBlock_.internal) {
        vorbis_block_clear(&vorbisBlock_);
    }

    if (vorbisDspState_.internal) {
        vorbis_synthesis_clear(&vorbisDspState_);
    }

    if (vorbisComment_.user_comments) {
        vorbis_comment_clear(&vorbisComment_);
    }

    if (vorbisInfo_.channels) {
        vorbis_info_clear(&vorbisInfo_);
    }

    memset(&vorbisInfo_, 0, sizeof(vorbisInfo_));
    memset(&vorbisComment_, 0, sizeof(vorbisComment_));
    memset(&vorbisDspState_, 0, sizeof(vorbisDspState_));
    memset(&vorbisBlock_, 0, sizeof(vorbisBlock_));
#endif // VORTEX_ENABLE_OGG
}

} // namespace vortex::core::fileio