#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <array>
#include <chrono>
#include <functional>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <queue>
#include <condition_variable>
#include <fstream>
#include "core/audio/multi_channel_engine.hpp"
#include "core/audio/session_manager.hpp"

namespace vortex {
namespace core {
namespace audio {

/**
 * Professional Audio Export/Import System
 * Supports high-quality audio file formats with professional metadata
 * and multi-channel support for professional audio applications
 */

enum class AudioFormat {
    WAV,                ///< WAV format (Microsoft)
    AIFF,               ///< AIFF format (Apple)
    FLAC,               ///< FLAC lossless compression
    ALAC,               ///< ALAC lossless compression
    MP3,                ///< MP3 compressed format
    AAC,                ///< AAC compressed format
    OGG_VORBIS,         ///< Ogg Vorbis compressed format
    OPUS,               ///< Opus compressed format
    WMA,                ///< Windows Media Audio
    DSD,                ///< Direct Stream Digital
    CAF,                ///< Core Audio Format
    RAW,                ///< Raw PCM data
    MATROSKA,           ///< Matroska container
    MIDI,               ///< MIDI file format
    PROJECT,            ///< Project file (DAW project)
    SESSION,            ///< Session file (audio session)
    METADATA_ONLY,      ///< Metadata only export
    CUSTOM              ///< Custom format
};

enum class AudioCodec {
    PCM_S16LE,           ///< 16-bit PCM little-endian
    PCM_S24LE,           ///< 24-bit PCM little-endian
    PCM_S32LE,           ///< 32-bit PCM little-endian
    PCM_FLOAT32,         ///< 32-bit float
    PCM_FLOAT64,         ///< 64-bit float
    PCM_ALAW,            ///< A-law PCM
    PCM_ULAW,            ///< μ-law PCM
    FLAC_8,              ///< FLAC 8-bit
    FLAC_16,             ///< FLAC 16-bit
    FLAC_24,             ///< FLAC 24-bit
    ALAC_16,             ///< ALAC 16-bit
    ALAC_24,             ///< ALAC 24-bit
    MP3_128,             ///< MP3 128 kbps
    MP3_192,             ///< MP3 192 kbps
    MP3_256,             ///< MP3 256 kbps
    MP3_320,             ///< MP3 320 kbps
    AAC_128,             ///< AAC 128 kbps
    AAC_192,             ///< AAC 192 kbps
    AAC_256,             ///< AAC 256 kbps
    AAC_320,             ///< AAC 320 kbps
    VORBIS_VBR,          ///< Vorbis VBR
    VORBIS_CBR,          ///< Vorbis CBR
    OPUS_VBR,            ///< Opus VBR
    OPUS_CBR,            ///< Opus CBR
    DSD64,               ///< DSD64 (1-bit 2.8224 MHz)
    DSD128,              ///< DSD128 (1-bit 5.6448 MHz)
    DSD256,              ///< DSD256 (1-bit 11.2896 MHz)
    DSD512,              ///< DSD512 (1-bit 22.5792 MHz)
    CUSTOM_CODEC         ///< Custom codec
};

enum class SampleRate {
    RATE_8000,           ///< 8 kHz
    RATE_11025,          ///< 11.025 kHz
    RATE_16000,          ///< 16 kHz
    RATE_22050,          ///< 22.05 kHz
    RATE_32000,          ///< 32 kHz
    RATE_44100,          ///< 44.1 kHz
    RATE_48000,          ///< 48 kHz
    RATE_88200,          ///< 88.2 kHz
    RATE_96000,          ///< 96 kHz
    RATE_176400,         ///< 176.4 kHz
    RATE_192000,         ///< 192 kHz
    RATE_352800,         ///< 352.8 kHz
    RATE_384000,         ///< 384 kHz
    RATE_705600,         ///< 705.6 kHz
    RATE_768000,         ///< 768 kHz
    CUSTOM_RATE          ///< Custom sample rate
};

enum class BitDepth {
    BIT_8,               ///< 8-bit
    BIT_16,              ///< 16-bit
    BIT_24,              ///< 24-bit
    BIT_32,              ///< 32-bit
    BIT_64,              ///< 64-bit
    CUSTOM_BIT           ///< Custom bit depth
};

enum class ChannelLayout {
    MONO,               ///< 1.0 - Mono
    STEREO,             ///< 2.0 - Stereo
    THREE_CHANNEL,       ///< 3.0 - Three channel
    FOUR_CHANNEL,        ///< 4.0 - Four channel
    FIVE_CHANNEL,        ///< 5.0 - Five channel
    FIVE_POINT_ONE,      ///< 5.1 - Five point one
    SIX_CHANNEL,         ///< 6.0 - Six channel
    SIX_POINT_ONE,       ///< 6.1 - Six point one
    SEVEN_CHANNEL,       ///< 7.0 - Seven channel
    SEVEN_POINT_ONE,     ///< 7.1 - Seven point one
    SEVEN_POINT_FOUR,    ///< 7.1.4 - Seven point one four
    NINE_POINT_ONE_FOUR,  ///< 9.1.4 - Nine point one four
    ELEVEN_POINT_ONE_FOUR, ///< 11.1.4 - Eleven point one four
    DOLBY_ATMOS,        ///< Dolby Atmos
    DTS_X,              ///< DTS:X
    Auro_3D,            ///< Auro-3D
    CUSTOM_LAYOUT       ///< Custom layout
};

enum class MetadataStandard {
    ID3V1,               ///< ID3v1 tags
    ID3V2,               ///< ID3v2 tags
    VORBIS,              ///< Vorbis comments
    RIFF,                ///< RIFF INFO chunk
    BWF,                 ///< Broadcast Wave Format
    CART,                ///< Cart chunk
    XMP,                 ///< XMP metadata
    IXML,                ///< iXML metadata
    BW64,                ///< BW64 metadata
    AES31,               ///< AES31 metadata
    EBU_TECH,            ///< EBU Technical metadata
    EBU_R128,            ///< EBU R128 loudness
    CUSTOM_STANDARD      ///< Custom standard
};

enum class ExportMode {
    FILE,                ///< Export to file
    STREAM,              ///< Export to stream
    BUFFER,              ///< Export to memory buffer
    DEVICE,              ///< Export to device
    NETWORK,             ///< Export to network
    PIPELINE,            ///< Export to processing pipeline
    MULTIPLE_FILES,      ///< Export to multiple files
    BATCH                ///< Batch export
};

enum class ImportMode {
    FILE,                ///< Import from file
    STREAM,              ///< Import from stream
    BUFFER,              ///< Import from memory buffer
    DEVICE,              ///< Import from device
    NETWORK,             ///< Import from network
    PIPELINE,            ///< Import from processing pipeline
    MULTIPLE_FILES,      ///< Import from multiple files
    BATCH                ///< Batch import
};

struct AudioFileInfo {
    std::string file_path;                    ///< File path
    std::string file_name;                    ///< File name
    AudioFormat format = AudioFormat::WAV;     ///< Audio format
    AudioCodec codec = AudioCodec::PCM_S16LE;   ///< Audio codec
    SampleRate sample_rate = SampleRate::RATE_44100; ///< Sample rate
    BitDepth bit_depth = BitDepth::BIT_16;      ///< Bit depth
    ChannelLayout channel_layout = ChannelLayout::STEREO; ///< Channel layout
    int channels = 2;                          ///< Number of channels
    int64_t file_size_bytes = 0;               ///< File size in bytes
    double duration_seconds = 0.0;             ///< Duration in seconds
    int64_t sample_count = 0;                   ///< Total sample count
    int bit_rate = 0;                          ///< Bit rate (bps)
    bool is_lossless = true;                   ///< Lossless codec
    bool is_encrypted = false;                 ///< File is encrypted
    std::chrono::system_clock::time_point creation_time; ///< File creation time
    std::chrono::system_clock::time_point modification_time; ///< File modification time
    std::string checksum;                      ///< File checksum (MD5/SHA)
    std::vector<uint8_t> thumbnail;            ///< Audio thumbnail/spectrogram
};

struct AudioMetadata {
    // Basic metadata
    std::string title;                          ///< Track title
    std::string artist;                         ///< Artist name
    std::string album;                          ///< Album name
    std::string genre;                          ///< Genre
    std::string year;                           ///< Year
    std::string track_number;                   ///< Track number
    std::string disc_number;                    ///< Disc number
    std::string comment;                        ///< Comment
    std::string copyright;                      ///< Copyright information
    std::string producer;                       ///< Producer
    std::string composer;                       ///< Composer
    std::string lyricist;                       ///< Lyricist
    std::string engineer;                       ///< Engineer
    std::string studio;                         ///< Recording studio

    // Technical metadata
    std::string equipment;                      ///< Equipment used
    std::string software;                       ///< Software used
    std::string hardware;                       ///< Hardware used
    std::string processing;                     ///< Processing applied

    // Professional metadata
    double loudness_lufs = -23.0;              ///< Integrated loudness (LUFS)
    double peak_level_dbtp = -1.0;             ///< True peak level (dBTP)
    double range_lu = 0.0;                     ///< Loudness range (LU)
    std::vector<double> per_channel_loudness;   ///< Per-channel loudness
    std::vector<double> per_channel_peaks;      ///< Per-channel peaks

    // Timing information
    std::chrono::milliseconds offset{0};        ///< Offset from beginning
    std::chrono::milliseconds length{0};         ///< Length
    std::vector<std::chrono::milliseconds> cue_points; ///< Cue points
    std::vector<std::chrono::milliseconds> markers; ///< Markers

    // BWF (Broadcast Wave Format) metadata
    std::string broadcast_date;                 ///< Broadcast date
    std::string originator;                     ///< Originator
    std::string originator_reference;          ///< Originator reference
    std::string umid;                           ///< Unique Material Identifier
    std::string coding_history;                 ///< Coding history

    // Custom metadata
    std::unordered_map<std::string, std::string> custom_tags; ///< Custom tags
    std::unordered_map<std::string, std::vector<uint8_t>> binary_data; ///< Binary data
    std::unordered_map<MetadataStandard, std::string> embedded_metadata; ///< Embedded metadata
};

struct ExportParameters {
    AudioFormat format = AudioFormat::WAV;       ///< Output format
    AudioCodec codec = AudioCodec::PCM_S16LE;     ///< Output codec
    SampleRate sample_rate = SampleRate::RATE_44100; ///< Output sample rate
    BitDepth bit_depth = BitDepth::BIT_16;        ///< Output bit depth
    ChannelLayout channel_layout = ChannelLayout::STEREO; ///< Output channel layout
    int channels = 2;                            ///< Number of output channels
    int quality_level = 5;                       ///< Quality level (1-10)
    int bit_rate = 0;                            ///< Target bit rate (0 for auto)
    bool normalize = false;                      ///< Normalize audio
    float normalize_level_dbfs = -1.0f;           ///< Normalization level
    bool dither = false;                         ///< Apply dithering
    bool apply_fade = false;                     ///< Apply fade in/out
    float fade_in_duration_seconds = 0.0f;       ///< Fade in duration
    float fade_out_duration_seconds = 0.0f;      ///< Fade out duration
    bool trim_silence = false;                  ///< Trim silence
    float silence_threshold_dbfs = -60.0f;       ///< Silence threshold
    bool include_metadata = true;                ///< Include metadata
    std::vector<MetadataStandard> metadata_standards; ///< Metadata standards
    std::string custom_file_extension;           ///< Custom file extension
    ExportMode mode = ExportMode::FILE;           ///< Export mode
    std::string output_path;                     ///< Output path
    std::string file_naming_pattern;             ///< File naming pattern
    bool overwrite_existing = false;             ///< Overwrite existing files
    bool create_directories = true;             ///< Create directories as needed
};

struct ImportParameters {
    ImportMode mode = ImportMode::FILE;          ///< Import mode
    std::string source_path;                     ///< Source path or identifier
    bool auto_detect_format = true;             ///< Auto-detect format
    bool auto_detect_metadata = true;            ///< Auto-detect metadata
    bool verify_checksum = false;                ///< Verify file checksum
    bool resample_if_needed = false;            ///< Resample if needed
    SampleRate target_sample_rate = SampleRate::CUSTOM_RATE; ///< Target sample rate
    BitDepth target_bit_depth = BitDepth::CUSTOM_BIT; ///< Target bit depth
    ChannelLayout target_channel_layout = ChannelLayout::CUSTOM_LAYOUT; ///< Target channel layout
    int target_channels = 0;                    ///< Target channels (0 = keep original)
    bool normalize_on_import = false;            ///< Normalize on import
    float normalize_level_dbfs = -1.0f;           ///< Normalization level
    bool split_channels = false;                 ///< Split to individual files
    bool merge_channels = false;                 ///< Merge channels
    std::string output_path;                     ///< Output directory
    std::string file_naming_pattern;             ///< File naming pattern
    bool overwrite_existing = false;             ///< Overwrite existing files
    bool create_directories = true;             ///< Create directories as needed
};

struct ExportProgress {
    uint64_t bytes_exported = 0;                 ///< Bytes exported so far
    uint64_t total_bytes = 0;                    ///< Total bytes to export
    double progress_percent = 0.0;               ///< Progress percentage (0-100)
    double processing_rate_mbps = 0.0;          ///< Processing rate (MB/s)
    std::chrono::steady_clock::time_point start_time; ///< Export start time
    std::chrono::steady_clock::time_point estimated_completion; ///< Estimated completion time
    std::string current_operation;               ///< Current operation description
    bool is_complete = false;                   ///< Export complete
    bool has_error = false;                      ///< Export has error
    std::string error_message;                   ///< Error message if any
};

struct ImportProgress {
    uint64_t bytes_imported = 0;                ///< Bytes imported so far
    uint64_t total_bytes = 0;                   ///< Total bytes to import
    double progress_percent = 0.0;               ///< Progress percentage (0-100)
    double processing_rate_mbps = 0.0;          ///< Processing rate (MB/s)
    std::chrono::steady_clock::time_point start_time; ///< Import start time
    std::chrono::steady_clock::time_point estimated_completion; ///< Estimated completion time
    std::string current_operation;               ///< Current operation description
    int files_processed = 0;                    ///< Number of files processed
    int total_files = 0;                        ///< Total number of files
    bool is_complete = false;                   ///< Import complete
    bool has_error = false;                      ///< Import has error
    std::string error_message;                   ///< Error message if any
};

using ProgressCallback = std::function<void(const ExportProgress& progress)>;
using ImportProgressCallback = std::function<void(const ImportProgress& progress)>;
using CompletionCallback = std::function<void(bool success, const std::string& message)>;

class AudioExporter {
public:
    AudioExporter();
    ~AudioExporter();

    /**
     * Initialize audio exporter
     * @return True if initialization successful
     */
    bool initialize();

    /**
     * Shutdown audio exporter
     */
    void shutdown();

    /**
     * Export audio file
     * @param input_buffer Input audio buffer
     * @param num_samples Number of samples
     * @param input_sample_rate Input sample rate
     * @param input_channels Input channels
     * @param parameters Export parameters
     * @param metadata Audio metadata
     * @param progress_callback Progress callback (optional)
     * @return True if export successful
     */
    bool exportAudio(const float* input_buffer, size_t num_samples, int input_sample_rate, int input_channels,
                     const ExportParameters& parameters, const AudioMetadata& metadata = {},
                     ProgressCallback progress_callback = nullptr);

    /**
     * Export audio to file
     * @param input_file Input file path
     * @param output_file Output file path
     * @param parameters Export parameters
     * @param include_metadata Include metadata from source file
     * @param progress_callback Progress callback (optional)
     * @return True if export successful
     */
    bool exportFile(const std::string& input_file, const std::string& output_file,
                     const ExportParameters& parameters, bool include_metadata = true,
                     ProgressCallback progress_callback = nullptr);

    /**
     * Export multiple files in batch
     * @param input_files List of input files
     * @param output_directory Output directory
     * @param parameters Export parameters
     * @param progress_callback Progress callback (optional)
     * @return True if batch export successful
     */
    bool exportBatch(const std::vector<std::string>& input_files, const std::string& output_directory,
                     const ExportParameters& parameters, ProgressCallback progress_callback = nullptr);

    /**
     * Export to stream
     * @param input_buffer Input audio buffer
     * @param num_samples Number of samples
     * @param input_sample_rate Input sample rate
     * @param input_channels Input channels
     * @param stream_output Stream output function
     * @param parameters Export parameters
     * @param metadata Audio metadata
     * @return True if export successful
     */
    bool exportToStream(const float* input_buffer, size_t num_samples, int input_sample_rate, int input_channels,
                        std::function<void(const uint8_t* data, size_t size)> stream_output,
                        const ExportParameters& parameters, const AudioMetadata& metadata = {});

    /**
     * Cancel current export
     */
    void cancelExport();

    /**
     * Get current export progress
     * @return Current export progress
     */
    ExportProgress getExportProgress() const;

    /**
     * Get supported export formats
     * @return List of supported formats
     */
    std::vector<AudioFormat> getSupportedFormats() const;

    /**
     * Get supported codecs for format
     * @param format Audio format
     * @return List of supported codecs
     */
    std::vector<AudioCodec> getSupportedCodecs(AudioFormat format) const;

    /**
     * Get default export parameters for format
     * @param format Audio format
     * @return Default parameters
     */
    ExportParameters getDefaultParameters(AudioFormat format) const;

    /**
     * Validate export parameters
     * @param parameters Parameters to validate
     * @return True if parameters are valid
     */
    bool validateParameters(const ExportParameters& parameters) const;

    /**
     * Estimate output file size
     * @param num_samples Number of samples
     * @param sample_rate Sample rate
     * @param channels Number of channels
     * @param parameters Export parameters
     * @return Estimated file size in bytes
     */
    size_t estimateOutputSize(size_t num_samples, int sample_rate, int channels,
                               const ExportParameters& parameters) const;

    /**
     * Get audio file information
     * @param file_path File path
     * @return File information
     */
    std::optional<AudioFileInfo> getAudioFileInfo(const std::string& file_path) const;

    /**
     * Read audio metadata
     * @param file_path File path
     * @return Audio metadata
     */
    std::optional<AudioMetadata> readAudioMetadata(const std::string& file_path) const;

private:
    // Internal export methods
    bool exportWAV(const float* input_buffer, size_t num_samples, int input_sample_rate, int input_channels,
                   const ExportParameters& parameters, const AudioMetadata& metadata);
    bool exportFLAC(const float* input_buffer, size_t num_samples, int input_sample_rate, int input_channels,
                     const ExportParameters& parameters, const AudioMetadata& metadata);
    bool exportMP3(const float* input_buffer, size_t num_samples, int input_sample_rate, int input_channels,
                   const ExportParameters& parameters, const AudioMetadata& metadata);
    bool exportAAC(const float* input_buffer, size_t num_samples, int input_sample_rate, int input_channels,
                   const ExportParameters& parameters, const AudioMetadata& metadata);
    bool exportOGG(const float* input_buffer, size_t num_samples, int input_sample_rate, int input_channels,
                   const ExportParameters& parameters, const AudioMetadata& metadata);
    bool exportOPUS(const float* input_buffer, size_t num_samples, int input_sample_rate, int input_channels,
                    const ExportParameters& parameters, const AudioMetadata& metadata);
    bool exportDSD(const float* input_buffer, size_t num_samples, int input_sample_rate, int input_channels,
                   const ExportParameters& parameters, const AudioMetadata& metadata);

    // Audio processing methods
    std::vector<float> processAudioForExport(const float* input_buffer, size_t num_samples, int input_sample_rate,
                                            int input_channels, const ExportParameters& parameters);
    void applyNormalizer(std::vector<float>& buffer, int channels, float target_level_dbfs);
    void applyFade(std::vector<float>& buffer, int channels, size_t num_samples,
                   float fade_in_duration, float fade_out_duration, int sample_rate);
    void applyDithering(std::vector<float>& buffer, BitDepth target_bit_depth);
    void trimSilence(std::vector<float>& buffer, int channels, float silence_threshold_dbfs);

    // Metadata methods
    void writeMetadata(std::ofstream& file, const AudioMetadata& metadata, const std::vector<MetadataStandard>& standards);
    std::vector<uint8_t> generateMetadataChunk(const AudioMetadata& metadata, MetadataStandard standard);

    // State management
    bool initialized_ = false;
    std::atomic<bool> export_cancelled_{false};
    std::atomic<bool> export_in_progress_{false};
    mutable std::mutex progress_mutex_;
    ExportProgress current_progress_;
    ProgressCallback progress_callback_;

    // Audio processing buffers
    std::vector<float> processing_buffer_;
    std::vector<float> temp_buffer_;
};

class AudioImporter {
public:
    AudioImporter();
    ~AudioImporter();

    /**
     * Initialize audio importer
     * @return True if initialization successful
     */
    bool initialize();

    /**
     * Shutdown audio importer
     */
    void shutdown();

    /**
     * Import audio file
     * @param file_path File path
     * @param parameters Import parameters
     * @param progress_callback Progress callback (optional)
     * @return True if import successful
     */
    std::vector<float> importAudio(const std::string& file_path, const ImportParameters& parameters,
                                   ImportProgressCallback progress_callback = nullptr);

    /**
     * Import audio file with metadata
     * @param file_path File path
     * @param parameters Import parameters
     * @param metadata Output metadata
     * @param progress_callback Progress callback (optional)
     * @return Audio buffer if successful
     */
    std::vector<float> importAudio(const std::string& file_path, const ImportParameters& parameters,
                                   AudioMetadata& metadata, ImportProgressCallback progress_callback = nullptr);

    /**
     * Import audio file info only
     * @param file_path File path
     * @return File information
     */
    std::optional<AudioFileInfo> getAudioFileInfo(const std::string& file_path);

    /**
     * Import audio metadata only
     * @param file_path File path
     * @return Audio metadata
     */
    std::optional<AudioMetadata> readAudioMetadata(const std::string& file_path);

    /**
     * Import multiple files in batch
     * @param input_files List of input files
     * @param output_directory Output directory
     * @param parameters Import parameters
     * @param progress_callback Progress callback (optional)
     * @return True if batch import successful
     */
    bool importBatch(const std::vector<std::string>& input_files, const std::string& output_directory,
                     const ImportParameters& parameters, ImportProgressCallback progress_callback = nullptr);

    /**
     * Import from stream
     * @param stream_input Stream input function
     * @param file_size Expected file size
     * @param parameters Import parameters
     * @return Audio buffer if successful
     */
    std::vector<float> importFromStream(std::function<std::vector<uint8_t>(size_t size)> stream_input,
                                         size_t file_size, const ImportParameters& parameters);

    /**
     * Cancel current import
     */
    void cancelImport();

    /**
     * Get current import progress
     * @return Current import progress
     */
    ImportProgress getImportProgress() const;

    /**
     * Get supported import formats
     * @return List of supported formats
     */
    std::vector<AudioFormat> getSupportedFormats() const;

    /**
     * Detect audio file format
     * @param file_path File path
     * @return Detected format
     */
    std::optional<AudioFormat> detectFormat(const std::string& file_path);

    /**
     * Validate audio file
     * @param file_path File path
     * @return True if file is valid
     */
    bool validateAudioFile(const std::string& file_path);

    /**
     * Generate audio thumbnail
     * @param file_path File path
     * @param width Thumbnail width
     * @param height Thumbnail height
     * @return Thumbnail data (RGB pixels)
     */
    std::vector<uint8_t> generateThumbnail(const std::string& file_path, int width, int height);

    /**
     * Analyze audio file
     * @param file_path File path
     * @return Analysis results (peaks, RMS, etc.)
     */
    std::unordered_map<std::string, double> analyzeAudio(const std::string& file_path);

private:
    // Internal import methods
    std::vector<float> importWAV(const std::string& file_path, const ImportParameters& parameters);
    std::vector<float> importAIFF(const std::string& file_path, const ImportParameters& parameters);
    std::vector<float> importFLAC(const std::string& file_path, const ImportParameters& parameters);
    std::vector<float> importMP3(const std::string& file_path, const ImportParameters& parameters);
    std::vector<float> importAAC(const std::string& file_path, const ImportParameters& parameters);
    std::vector<float> importOGG(const std::string& file_path, const ImportParameters& parameters);
    std::vector<float> importOPUS(const std::string& file_path, const ImportParameters& parameters);
    std::vector<float> importDSD(const std::string& file_path, const ImportParameters& parameters);

    // Audio processing methods
    std::vector<float> processImportedAudio(const std::vector<float>& input_buffer, int input_sample_rate,
                                            int input_channels, const ImportParameters& parameters);
    std::vector<float> resampleAudio(const std::vector<float>& input_buffer, int input_sample_rate,
                                     int output_sample_rate);
    std::vector<float> convertBitDepth(const std::vector<float>& input_buffer, BitDepth input_bit_depth,
                                        BitDepth output_bit_depth);
    std::vector<float> convertChannelLayout(const std::vector<float>& input_buffer, int input_channels,
                                           ChannelLayout input_layout, int output_channels,
                                           ChannelLayout output_layout);

    // Metadata methods
    AudioMetadata readMetadataFromWAV(const std::string& file_path);
    AudioMetadata readMetadataFromAIFF(const std::string& file_path);
    AudioMetadata readMetadataFromFLAC(const std::string& file_path);
    AudioMetadata readMetadataFromMP3(const std::string& file_path);
    AudioMetadata readMetadataFromAAC(const std::string& file_path);
    AudioMetadata readMetadataFromOGG(const std::string& file_path);

    // State management
    bool initialized_ = false;
    std::atomic<bool> import_cancelled_{false};
    std::atomic<bool> import_in_progress_{false};
    mutable std::mutex progress_mutex_;
    ImportProgress current_progress_;
    ImportProgressCallback progress_callback_;

    // Audio processing buffers
    std::vector<float> processing_buffer_;
    std::vector<float> temp_buffer_;
};

// Utility functions
namespace audio_export_import_utils {

    // Format utilities
    std::string formatToString(AudioFormat format);
    std::string codecToString(AudioCodec codec);
    std::string sampleRateToString(SampleRate sample_rate);
    std::string bitDepthToString(BitDepth bit_depth);
    std::string channelLayoutToString(ChannelLayout layout);
    std::string metadataStandardToString(MetadataStandard standard);

    AudioFormat stringToFormat(const std::string& str);
    AudioCodec stringToCodec(const std::string& str);
    SampleRate stringToSampleRate(const std::string& str);
    BitDepth stringToBitDepth(const std::string& str);
    ChannelLayout stringToChannelLayout(const std::string& str);

    // File utilities
    std::string getFileExtension(AudioFormat format);
    std::string generateFileName(const std::string& pattern, const AudioMetadata& metadata, int index = 0);
    bool createDirectoryPath(const std::string& path);
    std::string calculateFileChecksum(const std::string& file_path);
    bool verifyFileChecksum(const std::string& file_path, const std::string& checksum);

    // Audio utilities
    double calculateLUFS(const float* buffer, size_t samples, int sample_rate);
    double calculateTruePeak(const float* buffer, size_t samples, int sample_rate);
    double calculateLoudnessRange(const float* buffer, size_t samples, int sample_rate);
    std::vector<double> calculatePerChannelLUFS(const float* buffer, size_t samples, int channels, int sample_rate);
    std::vector<float> calculatePerChannelPeaks(const float* buffer, size_t samples, int channels);
    std::vector<double> analyzeFrequencyContent(const float* buffer, size_t samples, int sample_rate);

    // Conversion utilities
    std::vector<float> resampleLinear(const float* input, size_t input_samples, int input_rate, int output_rate);
    std::vector<float> resampleSinc(const float* input, size_t input_samples, int input_rate, int output_rate);
    std::vector<float> applyDither(const float* input, size_t samples, BitDepth target_bit_depth);
    std::vector<float> normalizeAudio(const float* input, size_t samples, float target_level_dbfs);
    void applyFadeIn(float* buffer, size_t samples, float duration_seconds, int sample_rate);
    void applyFadeOut(float* buffer, size_t samples, float duration_seconds, int sample_rate);

    // Quality utilities
    int calculateQualityLevel(AudioCodec codec, int bit_rate);
    int getOptimalBitRate(AudioCodec codec, AudioFormat format, BitDepth bit_depth, int sample_rate);
    bool isLosslessCodec(AudioCodec codec);
    double calculateCompressionRatio(AudioCodec codec, int bit_rate, int sample_rate, int channels);

    // Metadata utilities
    std::string formatDuration(double seconds);
    std::string formatFileSize(uint64_t bytes);
    std::string formatBitRate(int bit_rate);
    std::string formatTimestamp(const std::chrono::system_clock::time_point& timestamp);
    std::string generateUMID();

    // Validation utilities
    bool isValidAudioFile(const std::string& file_path);
    bool isValidAudioFormat(AudioFormat format);
    bool isValidAudioCodec(AudioCodec codec, AudioFormat format);
    bool isValidSampleRate(SampleRate sample_rate);
    bool isValidBitDepth(BitDepth bit_depth);

    // Analysis utilities
    std::vector<float> generateWaveform(const std::vector<float>& audio, int width, int height);
    std::vector<float> generateSpectrogram(const std::vector<float>& audio, int sample_rate, int width, int height);
    std::vector<uint8_t> generateThumbnail(const std::vector<float>& audio, int sample_rate, int width, int height);
}

} // namespace audio
} // namespace core
} // namespace vortex