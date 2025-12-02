#include "flac_decoder.hpp"
#include <cstring>
#include <algorithm>
#include <stdexcept>

#ifdef VORTEX_ENABLE_FLAC
#include <FLAC++/decoder.h>
#include <FLAC++/metadata.h>
#endif

namespace vortex::core::fileio {

#ifdef VORTEX_ENABLE_FLAC

class FLACDecoderImpl : public FLAC::Decoder::File {
public:
    FLACDecoderImpl() : audioData_(nullptr), metadata_(nullptr), initialized_(false) {}

    bool initialize(AudioData* audioData, AudioMetadata* metadata = nullptr) {
        audioData_ = audioData;
        metadata_ = metadata;
        initialized_ = true;
        return set_md5_checking(false);  // Disable MD5 for performance
    }

    void shutdown() {
        finish();
        initialized_ = false;
        audioData_ = nullptr;
        metadata_ = nullptr;
    }

protected:
    ::FLAC__StreamDecoderReadStatus read_callback(FLAC__byte buffer[], size_t *bytes) override {
        if (!file_.is_open() || file_.eof()) {
            *bytes = 0;
            return ::FLAC__STREAM_DECODER_READ_STATUS_END_OF_STREAM;
        }

        file_.read(reinterpret_cast<char*>(buffer), *bytes);
        size_t bytesRead = file_.gcount();
        *bytes = bytesRead;

        if (bytesRead == 0) {
            return ::FLAC__STREAM_DECODER_READ_STATUS_END_OF_STREAM;
        }

        return ::FLAC__STREAM_DECODER_READ_STATUS_CONTINUE;
    }

    ::FLAC__StreamDecoderSeekStatus seek_callback(FLAC__uint64 absolute_byte_offset) override {
        if (!file_.is_open()) {
            return ::FLAC__STREAM_DECODER_SEEK_STATUS_ERROR;
        }

        file_.seekg(absolute_byte_offset);
        if (file_.fail()) {
            return ::FLAC__STREAM_DECODER_SEEK_STATUS_ERROR;
        }

        return ::FLAC__STREAM_DECODER_SEEK_STATUS_OK;
    }

    ::FLAC__StreamDecoderTellStatus tell_callback(FLAC__uint64 *absolute_byte_offset) override {
        if (!file_.is_open()) {
            return ::FLAC__STREAM_DECODER_TELL_STATUS_ERROR;
        }

        *absolute_byte_offset = file_.tellg();
        return ::FLAC__STREAM_DECODER_TELL_STATUS_OK;
    }

    ::FLAC__StreamDecoderLengthStatus length_callback(FLAC__uint64 *stream_length) override {
        if (!file_.is_open()) {
            return ::FLAC__STREAM_DECODER_LENGTH_STATUS_ERROR;
        }

        auto currentPos = file_.tellg();
        file_.seekg(0, std::ios::end);
        *stream_length = file_.tellg();
        file_.seekg(currentPos);

        return ::FLAC__STREAM_DECODER_LENGTH_STATUS_OK;
    }

    bool eof_callback() override {
        return !file_.is_open() || file_.eof();
    }

    ::FLAC__StreamDecoderWriteStatus write_callback(const ::FLAC__Frame *frame, const FLAC__int32 *const buffer[]) override {
        if (!audioData_ || !initialized_) {
            return ::FLAC__STREAM_DECODER_WRITE_STATUS_ABORT;
        }

        uint32_t samplesPerChannel = frame->header.blocksize;
        uint32_t totalSamples = samplesPerChannel * frame->header.channels;

        // Resize output buffer if needed
        size_t currentSize = audioData_->data.size();
        size_t requiredSize = (audioData_->sampleIndex + totalSamples) * sizeof(float);
        if (requiredSize > currentSize) {
            audioData_->data.resize(requiredSize);
        }

        float* pcmData = reinterpret_cast<float*>(audioData_->data.data()) + audioData_->sampleIndex;

        // Convert FLAC samples to float
        const double scale = 1.0 / (1UL << (frame->header.bits_per_sample - 1));

        for (uint32_t channel = 0; channel < frame->header.channels; ++channel) {
            for (uint32_t sample = 0; sample < samplesPerChannel; ++sample) {
                uint32_t outputIndex = sample * frame->header.channels + channel;
                pcmData[outputIndex] = static_cast<float>(buffer[channel][sample] * scale);
            }
        }

        audioData_->sampleIndex += totalSamples;
        return ::FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
    }

    void metadata_callback(const ::FLAC__StreamMetadata *metadata) override {
        if (!metadata_ || !initialized_) {
            return;
        }

        switch (metadata->type) {
            case FLAC__METADATA_TYPE_STREAMINFO:
                if (audioData_) {
                    audioData_->sampleRate = static_cast<double>(metadata->data.stream_info.sample_rate);
                    audioData_->channels = static_cast<uint16_t>(metadata->data.stream_info.channels);
                    audioData_->bitDepth = static_cast<uint16_t>(metadata->data.stream_info.bits_per_sample);
                    audioData_->format = AudioFormat::FLAC;

                    // Pre-allocate buffer
                    uint32_t totalSamples = metadata->data.stream_info.total_samples;
                    if (totalSamples > 0) {
                        size_t bufferSize = totalSamples * audioData_->channels * sizeof(float);
                        audioData_->data.reserve(bufferSize);
                    }
                }

                if (metadata_) {
                    metadata_->sampleRate = metadata->data.stream_info.sample_rate;
                    metadata_->channels = metadata->data.stream_info.channels;
                    metadata_->bitDepth = metadata->data.stream_info.bits_per_sample;

                    if (metadata->data.stream_info.total_samples > 0) {
                        metadata_->duration = std::chrono::duration<double>(
                            static_cast<double>(metadata->data.stream_info.total_samples) /
                            (metadata->data.stream_info.sample_rate * metadata->data.stream_info.channels));
                    }

                    // Calculate bitrate (FLAC is lossless, so bitrate varies)
                    metadata_->bitrate = static_cast<uint32_t>(
                        metadata->data.stream_info.sample_rate *
                        metadata->data.stream_info.channels *
                        metadata->data.stream_info.bits_per_sample);
                }
                break;

            case FLAC__METADATA_TYPE_VORBIS_COMMENT:
                if (metadata_) {
                    parseVorbisComments(metadata->data.vorbis_comment);
                }
                break;

            default:
                // Ignore other metadata types for now
                break;
        }
    }

    void error_callback(::FLAC__StreamDecoderErrorStatus status) override {
        const char* errorString = FLAC__StreamDecoderErrorStatusString[status];
        Logger::error("FLAC decoder error: {}", errorString);
    }

public:
    bool openFile(const std::string& filePath) {
        file_.open(filePath, std::ios::binary);
        if (!file_.is_open()) {
            Logger::error("Cannot open FLAC file: {}", filePath);
            return false;
        }
        return true;
    }

    void closeFile() {
        if (file_.is_open()) {
            file_.close();
        }
    }

private:
    void parseVorbisComments(const FLAC__StreamMetadata_VorbisComment& comments) {
        if (!metadata_) {
            return;
        }

        for (uint32_t i = 0; i < comments.num_comments; ++i) {
            const FLAC__StreamMetadata_VorbisComment_Entry& entry = comments.comments[i];
            std::string comment(entry.entry, entry.length);

            // Parse Vorbis comment format: "KEY=value"
            size_t separatorPos = comment.find('=');
            if (separatorPos != std::string::npos) {
                std::string key = comment.substr(0, separatorPos);
                std::string value = comment.substr(separatorPos + 1);

                // Normalize key to uppercase
                std::transform(key.begin(), key.end(), key.begin(), ::toupper);

                // Map common Vorbis comment keys
                if (key == "TITLE") {
                    metadata_->title = value;
                } else if (key == "ARTIST") {
                    metadata_->artist = value;
                } else if (key == "ALBUM") {
                    metadata_->album = value;
                } else if (key == "DATE") {
                    try {
                        metadata_->year = static_cast<uint16_t>(std::stoi(value.substr(0, 4)));
                    } catch (...) {
                        // Invalid year format
                    }
                } else if (key == "GENRE") {
                    metadata_->genre = value;
                } else if (key == "TRACKNUMBER") {
                    try {
                        metadata_->track = static_cast<uint16_t>(std::stoi(value));
                    } catch (...) {
                        // Invalid track number
                    }
                }
            }
        }
    }

    std::ifstream file_;
    AudioData* audioData_;
    AudioMetadata* metadata_;
    bool initialized_;
};

#endif // VORTEX_ENABLE_FLAC

FLACDecoder::FLACDecoder() : initialized_(false) {}

FLACDecoder::~FLACDecoder() {
    shutdown();
}

bool FLACDecoder::initialize() {
    if (initialized_) {
        return true;
    }

#ifndef VORTEX_ENABLE_FLAC
    Logger::error("FLAC support not enabled in build");
    return false;
#else
    initialized_ = true;
    Logger::info("FLAC decoder initialized successfully");
    return true;
#endif
}

void FLACDecoder::shutdown() {
    if (!initialized_) {
        return;
    }

    initialized_ = false;
    Logger::info("FLAC decoder shutdown");
}

bool FLACDecoder::canDecode(const std::string& filePath) const {
    if (!initialized_) {
        return false;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Read FLAC signature
    uint8_t signature[4];
    file.read(reinterpret_cast<char*>(signature), 4);
    size_t bytesRead = file.gcount();

    file.close();

    if (bytesRead < 4) {
        return false;
    }

    // Check for FLAC signature "fLaC"
    return (signature[0] == 'f' && signature[1] == 'L' && signature[2] == 'a' && signature[3] == 'C');
}

std::optional<AudioData> FLACDecoder::decode(const std::string& filePath) {
    if (!initialized_) {
        Logger::error("FLAC decoder not initialized");
        return std::nullopt;
    }

#ifndef VORTEX_ENABLE_FLAC
    Logger::error("FLAC support not enabled in build");
    return std::nullopt;
#else

    Logger::info("Decoding FLAC file: {}", filePath);

    AudioData audioData;
    audioData.sampleIndex = 0;  // Initialize sample counter

    FLACDecoderImpl decoder;
    if (!decoder.initialize(&audioData)) {
        Logger::error("Failed to initialize FLAC decoder implementation");
        return std::nullopt;
    }

    if (!decoder.openFile(filePath)) {
        decoder.shutdown();
        return std::nullopt;
    }

    try {
        // Initialize decoder
        ::FLAC__StreamDecoderInitStatus initStatus = decoder.init();
        if (initStatus != FLAC__STREAM_DECODER_INIT_STATUS_OK) {
            Logger::error("Failed to initialize FLAC stream decoder: {}", initStatus);
            decoder.closeFile();
            decoder.shutdown();
            return std::nullopt;
        }

        // Process entire file
        if (!decoder.process_until_end_of_stream()) {
            Logger::error("FLAC decoding failed");
            decoder.closeFile();
            decoder.shutdown();
            return std::nullopt;
        }

        // Finalize audio data
        decoder.finish();

        Logger::info("FLAC decoded successfully: {} samples, {} channels, {:.2f} seconds",
                    audioData.sampleIndex / audioData.channels, audioData.channels,
                    static_cast<double>(audioData.sampleIndex / audioData.channels) / audioData.sampleRate);

        decoder.closeFile();
        decoder.shutdown();

        // Resize data to actual decoded size
        audioData.data.resize(audioData.sampleIndex * sizeof(float));

        return audioData;

    } catch (const std::exception& e) {
        Logger::error("Exception during FLAC decoding: {}", e.what());
        decoder.closeFile();
        decoder.shutdown();
        return std::nullopt;
    }

#endif // VORTEX_ENABLE_FLAC
}

bool FLACDecoder::extractMetadata(const std::string& filePath, AudioMetadata& metadata) {
    if (!initialized_) {
        return false;
    }

#ifndef VORTEX_ENABLE_FLAC
    Logger::error("FLAC support not enabled in build");
    return false;
#else

    Logger::info("Extracting FLAC metadata: {}", filePath);

    metadata.format = AudioFormat::FLAC;
    metadata.codec = "FLAC";

    FLACDecoderImpl decoder;
    if (!decoder.initialize(nullptr, &metadata)) {
        Logger::error("Failed to initialize FLAC decoder for metadata extraction");
        return false;
    }

    if (!decoder.openFile(filePath)) {
        decoder.shutdown();
        return false;
    }

    try {
        // Initialize decoder for metadata only
        ::FLAC__StreamDecoderInitStatus initStatus = decoder.init();
        if (initStatus != FLAC__STREAM_DECODER_INIT_STATUS_OK) {
            Logger::error("Failed to initialize FLAC stream decoder for metadata: {}", initStatus);
            decoder.closeFile();
            decoder.shutdown();
            return false;
        }

        // Process metadata only
        if (!decoder.process_until_end_of_metadata()) {
            Logger::error("Failed to process FLAC metadata");
            decoder.closeFile();
            decoder.shutdown();
            return false;
        }

        decoder.finish();

        Logger::info("FLAC metadata extracted successfully");
        decoder.closeFile();
        decoder.shutdown();

        return true;

    } catch (const std::exception& e) {
        Logger::error("Exception during FLAC metadata extraction: {}", e.what());
        decoder.closeFile();
        decoder.shutdown();
        return false;
    }

#endif // VORTEX_ENABLE_FLAC
}

} // namespace vortex::core::fileio