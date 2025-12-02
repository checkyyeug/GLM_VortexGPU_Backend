#pragma once

#include "audio_types.hpp"
#include "network_types.hpp"

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <variant>
#include <fstream>

namespace vortex {

/**
 * @brief Comprehensive audio metadata extractor
 *
 * Supports extraction from all supported audio formats including:
 * - ID3v1, ID3v2, ID3v2.3, ID3v2.4 tags
 * - Vorbis comments (OGG/FLAC)
 * - MP4/M4A atoms (iTunes metadata)
 * - RIFF INFO chunks (WAV/AIFF)
 * - DSD metadata (DSF/DFF)
 * - APEv2 tags
 * - Custom and extended metadata
 */
class MetadataExtractor {
public:
    MetadataExtractor();
    ~MetadataExtractor() = default;

    // Main extraction methods
    AudioMetadata extractMetadata(const std::string& filePath);
    bool extractMetadata(const std::string& filePath, AudioMetadata& metadata);

    // Format-specific extraction
    AudioMetadata extractFromMP3(const std::string& filePath);
    AudioMetadata extractFromWAV(const std::string& filePath);
    AudioMetadata extractFromFLAC(const std::string& filePath);
    AudioMetadata extractFromOGG(const std::string& filePath);
    AudioMetadata extractFromM4A(const std::string& filePath);
    AudioMetadata extractFromAAC(const std::string& filePath);
    AudioMetadata extractFromDSD(const std::string& filePath);
    AudioMetadata extractFromALAC(const std::string& filePath);
    AudioMetadata extractFromAIFF(const std::string& filePath);

    // Technical metadata extraction
    struct TechnicalMetadata {
        AudioFormat format = AudioFormat::UNKNOWN;
        std::string codec;
        std::string formatVersion;
        uint32_t bitrate = 0;           // bps
        uint32_t sampleRate = 0;         // Hz
        uint16_t bitDepth = 0;           // bits
        uint16_t channels = 0;           // number of audio channels
        Duration duration{0};           // seconds
        uint64_t totalSamples = 0;       // total number of PCM samples
        float dynamicRange = 0.0f;       // LUFS
        float peakLevel = 0.0f;          // dBFS
        float rmsLevel = 0.0f;           // dBFS
        float replayGain = 0.0f;         // dB
        std::vector<float> channelLevels; // peak levels per channel
        std::string encodingTool;       // encoder used
        std::string encodingSettings;   // encoder settings
        bool isVariableBitrate = false;
        uint32_t vbrMinBitrate = 0;
        uint32_t vbrMaxBitrate = 0;
        bool isLossless = false;
        float compressionRatio = 0.0f;
        std::chrono::system_clock::time_point creationDate;
    };

    TechnicalMetadata extractTechnicalMetadata(const std::string& filePath);

    // Album art extraction
    struct AlbumArt {
        std::vector<uint8_t> data;
        std::string mimeType;          // "image/jpeg", "image/png", etc.
        std::string description;
        std::string type;              // "front", "back", "artist", etc.
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t depth = 0;            // bits per pixel
        std::string encoding;
    };

    std::vector<AlbumArt> extractAlbumArt(const std::string& filePath);
    AlbumArt extractPrimaryAlbumArt(const std::string& filePath);

    // Extended metadata
    struct ExtendedMetadata {
        // Release information
        std::string recordLabel;
        std::string catalogNumber;
        std::string releaseDate;
        std::string originalReleaseDate;
        std::string recordingDate;
        uint16_t releaseYear = 0;
        uint16_t originalYear = 0;

        // Track information
        uint16_t trackNumber = 0;
        uint16_t totalTracks = 0;
        uint16_t discNumber = 0;
        uint16_t totalDiscs = 0;

        // Genre and mood
        std::string genre;
        std::string subgenre;
        std::string mood;
        std::string style;
        std::string theme;

        // Personnel
        std::string composer;
        std::string lyricist;
        std::string conductor;
        std::string orchestra;
        std::string ensemble;
        std::string producer;
        std::string engineer;
        std::string mixer;

        // Location and venue
        std::string location;
        std::string venue;
        std::string city;
        std::string country;
        std::string recordingLocation;

        // Copyright and licensing
        std::string copyright;
        std::string license;
        std::string publisher;
        std::string rights;
        bool isCopyrightProtected = false;
        std::string copyrightHolder;

        // ISRC and other identifiers
        std::string isrc;              // International Standard Recording Code
        std::string upc;               // Universal Product Code
        std::string ean;               // European Article Number
        std::string isbn;              // International Standard Book Number
        std::string grid;              // Global Release Identifier

        // Technical details
        std::string equipment;
        std::string software;
        std::string hardware;
        std::string processingInfo;

        // Custom fields
        std::map<std::string, std::string> customTags;
        std::vector<std::string> keywords;

        // Rating and popularity
        float rating = 0.0f;           // 0-5 stars
        float popularity = 0.0f;       // 0-100
        uint32_t playCount = 0;
        uint32_t skipCount = 0;
        std::string ratingSource;

        // Additional timestamps
        std::chrono::system_clock::time_point lastModified;
        std::chrono::system_clock::time_point lastPlayed;
        std::chrono::system_clock::time_point addedToLibrary;
    };

    ExtendedMetadata extractExtendedMetadata(const std::string& filePath);

    // Cue sheet support
    struct CueSheetEntry {
        uint8_t trackNumber = 0;
        std::string title;
        std::string performer;
        std::string songwriter;
        std::string genre;
        uint32_t index = 0;            // frame index
        Duration startTime{0};
        Duration duration{0};
        bool isTrack = true;
        std::string flags;
    };

    std::vector<CueSheetEntry> extractCueSheet(const std::string& filePath);
    bool hasCueSheet(const std::string& filePath);

    // ReplayGain information
    struct ReplayGainInfo {
        float trackGain = 0.0f;       // dB
        float trackPeak = 0.0f;       // dBFS
        float albumGain = 0.0f;       // dB
        float albumPeak = 0.0f;       // dBFS
        std::string reference loudness;
        bool hasTrackInfo = false;
        bool hasAlbumInfo = false;
    };

    ReplayGainInfo extractReplayGain(const std::string& filePath);

    // Audio analysis results
    struct AnalysisResult {
        float averageLevel = 0.0f;    // dBFS
        float peakLevel = 0.0f;       // dBFS
        float rmsLevel = 0.0f;        // dBFS
        float zeroCrossings = 0.0f;    // per second
        float dynamicRange = 0.0f;    // dB
        float crestFactor = 0.0f;     // dB
        float spectralCentroid = 0.0f; // Hz
        float spectralBandwidth = 0.0f; // Hz
        float harmonicDistortion = 0.0f; // percentage
        float noiseLevel = 0.0f;      // dBFS
        bool hasDCOffset = false;
        bool isClipped = false;
        std::vector<float> frequencyBands; // 1/3 octave bands
    };

    AnalysisResult analyzeAudioFile(const std::string& filePath);

    // Batch processing
    struct BatchResult {
        std::string filePath;
        bool success = false;
        AudioMetadata metadata;
        ExtendedMetadata extendedMetadata;
        std::vector<AlbumArt> albumArt;
        ReplayGainInfo replayGain;
        std::string errorMessage;
        std::chrono::milliseconds processingTime;
    };

    std::vector<BatchResult> extractMetadataBatch(const std::vector<std::string>& filePaths);
    std::future<std::vector<BatchResult>> extractMetadataBatchAsync(const std::vector<std::string>& filePaths);

    // Configuration
    struct ExtractorConfig {
        bool extractTechnicalMetadata = true;
        bool extractAlbumArt = true;
        bool extractExtendedMetadata = true;
        bool extractReplayGain = true;
        bool extractCueSheet = true;
        bool analyzeAudio = false;
        bool preserveOriginalArtwork = true;
        bool useDatabaseCache = true;
        uint32_t maxAlbumArtSize = 10 * 1024 * 1024; // 10MB
        std::string databasePath = "metadata_cache.db";
        bool enableAPITagLookup = false;
        std::string musicBrainzApiKey = "";
    };

    void setConfiguration(const ExtractorConfig& config);
    ExtractorConfig getConfiguration() const;

    // Database caching
    bool initializeDatabase(const std::string& dbPath);
    bool cacheMetadata(const std::string& filePath, const AudioMetadata& metadata);
    std::optional<AudioMetadata> getCachedMetadata(const std::string& filePath);
    bool clearCache();
    bool removeCacheEntry(const std::string& filePath);

    // API integration
    struct ExternalAPIResult {
        bool success = false;
        std::string artistBio;
        std::string albumReview;
        std::vector<std::string> similarArtists;
        std::vector<std::string> genres;
        std::string coverArtUrl;
        std::map<std::string, std::string> additionalInfo;
        std::string errorMessage;
    };

    ExternalAPIResult queryMusicBrainz(const std::string& artist, const std::string& album);
    ExternalAPIResult queryDiscogs(const std::string& artist, const std::string& album);

    // Validation and cleaning
    struct ValidationResult {
        bool isValid = true;
        std::vector<std::string> warnings;
        std::vector<std::string> errors;
        std::map<std::string, std::string> corrections;
    };

    ValidationResult validateMetadata(const AudioMetadata& metadata);
    AudioMetadata cleanMetadata(const AudioMetadata& metadata);

    // Utility functions
    static std::string formatDuration(Duration duration);
    static std::string formatBitrate(uint32_t bitrate);
    static std::string formatSampleRate(uint32_t sampleRate);
    static std::string formatChannels(uint16_t channels);
    static std::string formatDynamicRange(float range);
    static std::string formatLevel(float level);
    static std::string formatDate(std::chrono::system_clock::time_point date);

    // Error handling
    std::string getLastError() const;
    bool hasErrors() const;

private:
    // Format-specific parsers
    bool parseID3v2Tags(const std::string& filePath, AudioMetadata& metadata, ExtendedMetadata& extended);
    bool parseID3v1Tags(const std::string& filePath, AudioMetadata& metadata);
    bool parseVorbisComments(const std::string& filePath, AudioMetadata& metadata, ExtendedMetadata& extended);
    bool parseMP4Atoms(const std::string& filePath, AudioMetadata& metadata, ExtendedMetadata& extended);
    bool parseRIFFInfo(const std::string& filePath, AudioMetadata& metadata, ExtendedMetadata& extended);
    bool parseDSDMetadata(const std::string& filePath, AudioMetadata& metadata, ExtendedMetadata& extended);

    // ID3v2 helpers
    struct ID3v2Header {
        char identifier[3];     // "ID3"
        uint8_t version[2];      // major, minor version
        uint8_t flags;
        uint32_t size;           // unsynchronized safe integer
    };

    struct ID3v2Frame {
        char frameId[4];        // frame identifier
        uint32_t size;           // frame size
        uint16_t flags;          // frame flags
    };

    bool parseID3v2Header(const std::vector<uint8_t>& data, ID3v2Header& header);
    bool parseID3v2Frame(const std::vector<uint8_t>& frameData, std::string& frameId, std::vector<uint8_t>& frameContent);
    std::string decodeID3v2Text(const std::vector<uint8_t>& data, bool isUnicode);
    std::map<std::string, std::string> parseID3v2TextInformation(const std::vector<uint8_t>& data);

    // MP4 atom helpers
    struct MP4Atom {
        std::string type;
        uint64_t size;
        uint64_t offset;
        std::vector<std::unique_ptr<MP4Atom>> children;
        std::vector<uint8_t> data;
    };

    std::unique_ptr<MP4Atom> parseMP4Atoms(std::ifstream& file, uint64_t startOffset = 0);
    std::string extractMP4Text(const std::vector<uint8_t>& data);
    std::vector<uint8_t> extractMP4CoverArt(const std::vector<uint8_t>& data);

    // File reading helpers
    std::vector<uint8_t> readFileHeader(const std::string& filePath, size_t bytes = 1024);
    std::vector<uint8_t> readFileSection(const std::string& filePath, uint64_t offset, size_t size);
    std::string readTextFile(const std::string& filePath);

    // Utility functions
    std::string extractFirstNonEmptyField(const std::map<std::string, std::string>& fields);
    void trimWhitespace(std::string& str);
    bool isUTF8(const std::vector<uint8_t>& data);
    std::string ensureUTF8(const std::string& str);
    std::vector<uint8_t> base64Decode(const std::string& base64);

    // Error handling
    mutable std::string lastError_;
    void setError(const std::string& error) const;

    // Configuration
    ExtractorConfig config_;

    // Database (if implemented)
    bool databaseInitialized_ = false;
};

/**
 * @brief Metadata validation and correction utilities
 */
class MetadataValidator {
public:
    struct ValidationRule {
        std::string field;
        bool required;
        std::function<bool(const std::string&)> validator;
        std::string correction;
    };

    static bool validateField(const std::string& value, const std::string& field);
    static std::string correctField(const std::string& value, const std::string& field);
    static std::vector<ValidationRule> getDefaultValidationRules();

private:
    static bool isValidTitle(const std::string& title);
    static bool isValidArtist(const std::string& artist);
    static bool isValidYear(const std::string& year);
    static bool isValidTrackNumber(const std::string& track);
    static bool isValidGenre(const std::string& genre);
};

} // namespace vortex