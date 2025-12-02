#pragma once

#include "../../audio_types.hpp"
#include "../../network_types.hpp"

#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace vortex {

/**
 * @brief HTTP endpoint for audio file upload and processing
 *
 * Supports multi-part form data uploads with:
 * - File size validation (up to 1GB)
 * - Format validation and metadata extraction
 * - Progress tracking with WebSocket updates
 * - Background processing with GPU acceleration
 * - Duplicate file detection
 * - Automatic format conversion
 */
class AudioUploadEndpoint {
public:
    struct UploadRequest {
        std::string filename;
        std::string contentType;
        std::vector<uint8_t> fileData;
        std::map<std::string, std::string> metadata;
        std::string clientSessionId;
        bool enableGPUProcessing = true;
        bool generatePreviews = true;
        bool extractAlbumArt = true;
        ProcessingPriority priority = ProcessingPriority::NORMAL;
    };

    struct UploadResponse {
        std::string fileId;
        std::string filename;
        AudioFormat detectedFormat;
        AudioMetadata extractedMetadata;
        ProcessingStatus status;
        float progress = 0.0f;
        uint64_t fileSize = 0;
        float estimatedProcessingTime = 0.0f;
        std::vector<std::string> warnings;
        bool success = false;
        std::string errorMessage;
    };

    struct UploadProgress {
        std::string fileId;
        ProcessingStatus status;
        float progress = 0.0f;
        std::string currentStage;
        float stageProgress = 0.0f;
        uint64_t bytesProcessed = 0;
        uint64_t totalBytes = 0;
        float estimatedTimeRemaining = 0.0f;
        GPUStatus gpuStatus;
    };

    AudioUploadEndpoint();
    ~AudioUploadEndpoint();

    // Initialization
    bool initialize();
    void shutdown();
    bool isInitialized() const;

    // Main upload endpoint
    HTTPResponse handleUpload(const HTTPRequest& request);

    // Upload status and progress
    HTTPResponse handleUploadStatus(const HTTPRequest& request);
    HTTPResponse handleUploadProgress(const HTTPRequest& request);

    // File management
    HTTPResponse handleFileList(const HTTPRequest& request);
    HTTPResponse handleFileDelete(const HTTPRequest& request);
    HTTPResponse handleFileMetadata(const HTTPRequest& request);

    // Batch operations
    HTTPResponse handleBatchUpload(const HTTPRequest& request);
    HTTPResponse handleBatchStatus(const HTTPRequest& request);

    // Advanced upload features
    HTTPResponse handleResumeUpload(const HTTPRequest& request);
    HTTPResponse handleCancelUpload(const HTTPRequest& request);
    HTTPResponse handleDuplicateCheck(const HTTPRequest& request);

    // Configuration
    struct UploadConfig {
        uint64_t maxFileSize = 1024 * 1024 * 1024; // 1GB
        size_t maxConcurrentUploads = 10;
        uint32_t uploadTimeout = 300; // seconds
        std::vector<AudioFormat> supportedFormats;
        std::string uploadDirectory = "uploads/";
        std::string tempDirectory = "temp/";
        bool enableChunkedUploads = true;
        bool enableCompression = true;
        bool enableVirusScanning = false;
        bool enableFormatValidation = true;
        bool autoGeneratePreviews = true;
        ProcessingPriority defaultPriority = ProcessingPriority::NORMAL;
    };

    void setConfiguration(const UploadConfig& config);
    UploadConfig getConfiguration() const;

    // Event callbacks
    using UploadStartCallback = std::function<void(const UploadRequest&)>;
    using UploadProgressCallback = std::function<void(const UploadProgress&)>;
    using UploadCompleteCallback = std::function<void(const UploadResponse&)>;
    using UploadErrorCallback = std::function<void(const std::string&, const std::string&)>;

    void setUploadStartCallback(UploadStartCallback callback);
    void setUploadProgressCallback(UploadProgressCallback callback);
    void setUploadCompleteCallback(UploadCompleteCallback callback);
    void setUploadErrorCallback(UploadErrorCallback callback);

private:
    // Request parsing
    UploadRequest parseUploadRequest(const HTTPRequest& request);
    bool validateUploadRequest(const UploadRequest& request);
    std::string generateFileId();
    std::string generateTempFilePath(const std::string& fileId, const std::string& extension);

    // File processing
    bool saveUploadedFile(const UploadRequest& request, const std::string& filePath);
    UploadResponse processAudioFile(const std::string& filePath, const UploadRequest& request);
    void startBackgroundProcessing(const std::string& fileId, const std::string& filePath,
                                  const UploadRequest& request);

    // Background processing pipeline
    struct ProcessingTask {
        std::string fileId;
        std::string filePath;
        UploadRequest request;
        std::chrono::steady_clock::time_point startTime;
        bool isCancelled = false;
    };

    void processingThread();
    void updateProcessingProgress(const std::string& fileId, ProcessingStatus status,
                                  float progress, const std::string& stage = "");
    void completeProcessing(const std::string& fileId, const UploadResponse& response);

    // Format validation and metadata extraction
    bool validateAudioFormat(const std::string& filePath);
    AudioMetadata extractFileMetadata(const std::string& filePath);
    std::vector<uint8_t> extractAlbumArt(const std::string& filePath);

    // Duplicate detection
    std::string findDuplicateFile(const std::string& filePath, const AudioMetadata& metadata);
    bool areFilesIdentical(const std::string& file1, const std::string& file2);

    // Chunked upload support
    struct ChunkInfo {
        std::string fileId;
        std::string chunkId;
        uint32_t chunkNumber = 0;
        uint32_t totalChunks = 0;
        size_t chunkSize = 0;
        std::vector<uint8_t> chunkData;
        std::string checksum;
    };

    HTTPResponse handleChunkUpload(const HTTPRequest& request);
    bool assembleChunkedFile(const std::string& fileId);
    std::string calculateFileChecksum(const std::string& filePath);

    // Security and validation
    bool validateFileSignature(const std::string& filePath, AudioFormat expectedFormat);
    bool scanForMalware(const std::string& filePath);
    bool sanitizeFilename(std::string& filename);

    // Progress tracking
    struct UploadProgressInfo {
        std::string fileId;
        std::atomic<float> progress{0.0f};
        std::atomic<ProcessingStatus> status{ProcessingStatus::IDLE};
        std::string currentStage;
        std::atomic<float> stageProgress{0.0f};
        std::atomic<uint64_t> bytesProcessed{0};
        std::atomic<uint64_t> totalBytes{0};
        std::chrono::steady_clock::time_point startTime;
        std::chrono::steady_clock::time_point lastUpdate;
    };

    std::unordered_map<std::string, std::unique_ptr<UploadProgressInfo>> uploadProgress_;
    mutable std::mutex progressMutex_;

    // File management
    struct FileInfo {
        std::string fileId;
        std::string filePath;
        std::string originalName;
        AudioFormat format;
        AudioMetadata metadata;
        uint64_t fileSize;
        std::chrono::steady_clock::time_point uploadTime;
        bool isProcessing = false;
        ProcessingStatus processingStatus = ProcessingStatus::IDLE;
    };

    std::unordered_map<std::string, FileInfo> uploadedFiles_;
    mutable std::mutex filesMutex_;

    // Background processing
    std::unique_ptr<std::thread> processingThread_;
    std::queue<std::unique_ptr<ProcessingTask>> processingQueue_;
    std::mutex queueMutex_;
    std::condition_variable queueCondition_;
    std::atomic<bool> shouldShutdown_{false};

    // Configuration
    UploadConfig config_;

    // Callbacks
    UploadStartCallback uploadStartCallback_;
    UploadProgressCallback uploadProgressCallback_;
    UploadCompleteCallback uploadCompleteCallback_;
    UploadErrorCallback uploadErrorCallback_;

    // Statistics
    struct Statistics {
        uint64_t totalUploads = 0;
        uint64_t successfulUploads = 0;
        uint64_t failedUploads = 0;
        uint64_t totalBytesUploaded = 0;
        double averageUploadTime = 0.0;
        std::chrono::steady_clock::time_point lastUpload;
        std::map<AudioFormat, uint64_t> formatCounts;
        std::map<ProcessingStatus, uint64_t> statusCounts;
    };

    Statistics statistics_;
    mutable std::mutex statsMutex_;

    // Error handling
    mutable std::string lastError_;
    void setError(const std::string& error) const;

    // Utility methods
    std::string formatFileSize(uint64_t bytes) const;
    std::string formatDuration(double seconds) const;
    bool isValidFilename(const std::string& filename) const;
    std::string sanitizeExtension(const std::string& filename) const;

    // HTTP response builders
    HTTPResponse createSuccessResponse(const UploadResponse& uploadResponse);
    HTTPResponse createErrorResponse(const std::string& message, int statusCode = 400);
    HTTPResponse createProgressResponse(const std::string& fileId);

    // Cleanup
    void cleanupTempFiles();
    void cancelProcessing(const std::string& fileId);
    void removeFile(const std::string& fileId);
};

/**
 * @brief File upload utilities and helpers
 */
class AudioUploadUtils {
public:
    // File type detection
    static AudioFormat detectAudioFormat(const std::vector<uint8_t>& fileData);
    static std::string getAudioMimeType(AudioFormat format);
    static std::vector<std::string> getAudioExtensions(AudioFormat format);

    // File validation
    static bool isValidAudioFile(const std::string& filePath);
    static bool isValidAudioData(const std::vector<uint8_t>& fileData, AudioFormat format);
    static uint64_t calculateAudioDuration(const std::string& filePath, AudioFormat format);

    // File processing
    static bool generateAudioPreview(const std::string& inputPath, const std::string& outputPath,
                                     uint32_t sampleRate = 22050, uint32_t duration = 30);
    static bool convertAudioFormat(const std::string& inputPath, const std::string& outputPath,
                                  AudioFormat targetFormat, uint32_t bitrate = 0);

    // Metadata utilities
    static std::map<std::string, std::string> extractID3Tags(const std::string& filePath);
    static bool embedMetadata(const std::string& filePath, const std::map<std::string, std::string>& tags);
    static std::vector<uint8_t> extractEmbeddedAlbumArt(const std::string& filePath);
    static bool embedAlbumArt(const std::string& filePath, const std::vector<uint8_t>& albumArt);

    // File compression
    static std::vector<uint8_t> compressAudioData(const std::vector<uint8_t>& data);
    static std::vector<uint8_t> decompressAudioData(const std::vector<uint8_t>& compressedData);
};

} // namespace vortex