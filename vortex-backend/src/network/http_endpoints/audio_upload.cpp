#include "audio_upload.hpp"
#include "../webserver.hpp"
#include "../protocol/binary_protocol.hpp"
#include "../../core/fileio/audio_file_loader.hpp"
#include "../../core/fileio/format_detector.hpp"
#include "../../system/logger.hpp"
#include "../../utils/thread_pool.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <future>

namespace vortex {

AudioUploadEndpoint::AudioUploadEndpoint() {
    // Initialize supported formats
    config_.supportedFormats = {
        AudioFormat::MP3, AudioFormat::WAV, AudioFormat::FLAC, AudioFormat::ALAC,
        AudioFormat::AAC, AudioFormat::OGG, AudioFormat::M4A, AudioFormat::DSF,
        AudioFormat::DFF, AudioFormat::DSD64, AudioFormat::DSD128, AudioFormat::DSD256,
        AudioFormat::DSD512, AudioFormat::DSD1024
    };
}

AudioUploadEndpoint::~AudioUploadEndpoint() {
    shutdown();
}

bool AudioUploadEndpoint::initialize() {
    try {
        Logger::info("Initializing Audio Upload Endpoint");

        // Create upload directories if they don't exist
        std::filesystem::create_directories(config_.uploadDirectory);
        std::filesystem::create_directories(config_.tempDirectory);

        // Start background processing thread
        processingThread_ = std::make_unique<std::thread>(&AudioUploadEndpoint::processingThread, this);

        Logger::info("Audio Upload Endpoint initialized successfully");
        return true;

    } catch (const std::exception& e) {
        Logger::error("Failed to initialize Audio Upload Endpoint: {}", e.what());
        return false;
    }
}

void AudioUploadEndpoint::shutdown() {
    if (processingThread_) {
        shouldShutdown_.store(true);
        queueCondition_.notify_all();

        if (processingThread_->joinable()) {
            processingThread_->join();
        }
        processingThread_.reset();
    }

    // Cleanup temporary files
    cleanupTempFiles();

    Logger::info("Audio Upload Endpoint shutdown complete");
}

bool AudioUploadEndpoint::isInitialized() const {
    return processingThread_ != nullptr;
}

HTTPResponse AudioUploadEndpoint::handleUpload(const HTTPRequest& request) {
    try {
        Logger::info("Handling audio file upload request from {}", request.clientIP);

        // Parse upload request
        UploadRequest uploadRequest = parseUploadRequest(request);
        if (!validateUploadRequest(uploadRequest)) {
            return createErrorResponse("Invalid upload request", 400);
        }

        // Trigger upload start callback
        if (uploadStartCallback_) {
            uploadStartCallback_(uploadRequest);
        }

        // Generate file ID and save file
        std::string fileId = generateFileId();
        std::string extension = sanitizeExtension(uploadRequest.filename);
        std::string tempFilePath = generateTempFilePath(fileId, extension);

        if (!saveUploadedFile(uploadRequest, tempFilePath)) {
            return createErrorResponse("Failed to save uploaded file", 500);
        }

        // Update statistics
        {
            std::lock_guard<std::mutex> lock(statsMutex_);
            statistics_.totalUploads++;
            statistics_.totalBytesUploaded += uploadRequest.fileData.size();
            statistics_.lastUpload = std::chrono::steady_clock::now();
        }

        // Start processing in background
        startBackgroundProcessing(fileId, tempFilePath, uploadRequest);

        // Create initial response
        UploadResponse response;
        response.fileId = fileId;
        response.filename = uploadRequest.filename;
        response.status = ProcessingStatus::LOADING;
        response.progress = 0.0f;
        response.fileSize = uploadRequest.fileData.size();
        response.success = true;

        // Store file info
        {
            std::lock_guard<std::mutex> lock(filesMutex_);
            FileInfo fileInfo;
            fileInfo.fileId = fileId;
            fileInfo.filePath = tempFilePath;
            fileInfo.originalName = uploadRequest.filename;
            fileInfo.fileSize = uploadRequest.fileData.size();
            fileInfo.uploadTime = std::chrono::steady_clock::now();
            fileInfo.isProcessing = true;
            fileInfo.processingStatus = ProcessingStatus::LOADING;

            uploadedFiles_[fileId] = fileInfo;
        }

        // Initialize progress tracking
        {
            std::lock_guard<std::mutex> lock(progressMutex_);
            auto progressInfo = std::make_unique<UploadProgressInfo>();
            progressInfo->fileId = fileId;
            progressInfo->status.store(ProcessingStatus::LOADING);
            progressInfo->startTime = std::chrono::steady_clock::now();
            progressInfo->totalBytes.store(uploadRequest.fileData.size());
            progressInfo->bytesProcessed.store(uploadRequest.fileData.size());
            uploadProgress_[fileId] = std::move(progressInfo);
        }

        return createSuccessResponse(response);

    } catch (const std::exception& e) {
        Logger::error("Error handling upload request: {}", e.what());
        return createErrorResponse("Internal server error", 500);
    }
}

HTTPResponse AudioUploadEndpoint::handleUploadStatus(const HTTPRequest& request) {
    try {
        // Extract file ID from URL path
        std::string fileId;
        if (request.path.find("/api/audio/upload/") == 0) {
            fileId = request.path.substr(std::string("/api/audio/upload/").length());
        }

        if (fileId.empty()) {
            return createErrorResponse("File ID not specified", 400);
        }

        // Get file info
        FileInfo fileInfo;
        {
            std::lock_guard<std::mutex> lock(filesMutex_);
            auto it = uploadedFiles_.find(fileId);
            if (it == uploadedFiles_.end()) {
                return createErrorResponse("File not found", 404);
            }
            fileInfo = it->second;
        }

        // Create response
        UploadResponse response;
        response.fileId = fileId;
        response.filename = fileInfo.originalName;
        response.status = fileInfo.processingStatus;

        // Get progress info
        {
            std::lock_guard<std::mutex> lock(progressMutex_);
            auto it = uploadProgress_.find(fileId);
            if (it != uploadProgress_.end()) {
                response.progress = it->second->progress.load();
                response.estimatedProcessingTime = calculateEstimatedTimeRemaining(fileId);
            }
        }

        response.fileSize = fileInfo.fileSize;
        response.success = true;

        return createSuccessResponse(response);

    } catch (const std::exception& e) {
        Logger::error("Error handling upload status request: {}", e.what());
        return createErrorResponse("Internal server error", 500);
    }
}

HTTPResponse AudioUploadEndpoint::handleUploadProgress(const HTTPRequest& request) {
    try {
        // Parse JSON body for multiple file IDs
        std::vector<std::string> fileIds;
        if (!request.body.empty()) {
            rapidjson::Document doc;
            doc.Parse(request.body.c_str());
            if (!doc.HasParseError() && doc.HasMember("fileIds") && doc["fileIds"].IsArray()) {
                for (const auto& id : doc["fileIds"].GetArray()) {
                    if (id.IsString()) {
                        fileIds.push_back(id.GetString());
                    }
                }
            }
        }

        // Build progress response
        std::stringstream json;
        json << "{\"progress\":[";

        bool first = true;
        for (const std::string& fileId : fileIds) {
            if (!first) {
                json << ",";
            }
            first = false;

            UploadProgressInfo* progressInfo = nullptr;
            {
                std::lock_guard<std::mutex> lock(progressMutex_);
                auto it = uploadProgress_.find(fileId);
                if (it != uploadProgress_.end()) {
                    progressInfo = it->second.get();
                }
            }

            if (progressInfo) {
                json << "{"
                     << "\"fileId\":\"" << fileId << "\","
                     << "\"status\":\"" << statusToString(progressInfo->status.load()) << "\","
                     << "\"progress\":" << progressInfo->progress.load() << ","
                     << "\"currentStage\":\"" << progressInfo->currentStage << "\","
                     << "\"stageProgress\":" << progressInfo->stageProgress.load() << ","
                     << "\"bytesProcessed\":" << progressInfo->bytesProcessed.load() << ","
                     << "\"totalBytes\":" << progressInfo->totalBytes.load() << ","
                     << "\"estimatedTimeRemaining\":" << calculateEstimatedTimeRemaining(fileId)
                     << "}";
            } else {
                json << "{\"fileId\":\"" << fileId << "\",\"status\":\"not_found\"}";
            }
        }

        json << "]}";

        HTTPResponse response;
        response.status = HttpStatus::OK;
        response.body = json.str();
        response.contentType = "application/json";

        return response;

    } catch (const std::exception& e) {
        Logger::error("Error handling upload progress request: {}", e.what());
        return createErrorResponse("Internal server error", 500);
    }
}

AudioUploadEndpoint::UploadRequest AudioUploadEndpoint::parseUploadRequest(const HTTPRequest& request) {
    UploadRequest uploadRequest;

    // Parse form data
    if (request.headers.find("Content-Type") != request.headers.end() &&
        request.headers.at("Content-Type").find("multipart/form-data") != std::string::npos) {

        // For this implementation, we'll assume the file data is in request.body
        // In a real implementation, you'd parse multipart form data properly
        uploadRequest.fileData.assign(request.body.begin(), request.body.end());

        // Extract filename from content-disposition header or use default
        if (request.headers.find("Content-Disposition") != request.headers.end()) {
            std::string contentDisposition = request.headers.at("Content-Disposition");
            size_t pos = contentDisposition.find("filename=\"");
            if (pos != std::string::npos) {
                size_t start = pos + 10;
                size_t end = contentDisposition.find("\"", start);
                if (end != std::string::npos) {
                    uploadRequest.filename = contentDisposition.substr(start, end - start);
                }
            }
        }

        if (uploadRequest.filename.empty()) {
            uploadRequest.filename = "uploaded_file.bin";
        }

        uploadRequest.contentType = request.headers.find("Content-Type") != request.headers.end() ?
                                 request.headers.at("Content-Type") : "application/octet-stream";

        // Extract metadata from other headers or query parameters
        if (!request.queryParameters.empty()) {
            uploadRequest.metadata = request.queryParameters;
        }

        // Set default values
        uploadRequest.enableGPUProcessing = true;
        uploadRequest.generatePreviews = true;
        uploadRequest.extractAlbumArt = true;
        uploadRequest.priority = ProcessingPriority::NORMAL;

    } else {
        // JSON-based upload (for smaller files or metadata)
        rapidjson::Document doc;
        doc.Parse(request.body.c_str());

        if (doc.HasMember("filename") && doc["filename"].IsString()) {
            uploadRequest.filename = doc["filename"].GetString();
        }

        if (doc.HasMember("data") && doc["data"].IsString()) {
            // Assume base64 encoded data for JSON uploads
            std::string base64Data = doc["data"].GetString();
            uploadRequest.fileData = base64Decode(base64Data);
        }

        if (doc.HasMember("metadata") && doc["metadata"].IsObject()) {
            // Parse metadata object
            const auto& metadataObj = doc["metadata"].GetObject();
            for (auto it = metadataObj.MemberBegin(); it != metadataObj.MemberEnd(); ++it) {
                if (it->value.IsString()) {
                    uploadRequest.metadata[it->name.GetString()] = it->value.GetString();
                }
            }
        }
    }

    return uploadRequest;
}

bool AudioUploadEndpoint::validateUploadRequest(const UploadRequest& request) {
    // Check file size
    if (request.fileData.empty()) {
        Logger::warn("Empty file data in upload request");
        return false;
    }

    if (request.fileData.size() > config_.maxFileSize) {
        Logger::warn("File size {} exceeds maximum {}", request.fileData.size(), config_.maxFileSize);
        return false;
    }

    // Check filename
    if (request.filename.empty() || !isValidFilename(request.filename)) {
        Logger::warn("Invalid filename in upload request: {}", request.filename);
        return false;
    }

    // Validate file format by checking magic bytes
    std::vector<uint8_t> fileHeader(request.fileData.begin(),
                                    request.fileData.begin() + std::min<size_t>(request.fileData.size(), 32));
    AudioFormat detectedFormat = AudioUploadUtils::detectAudioFormat(fileHeader);

    bool formatSupported = std::find(config_.supportedFormats.begin(),
                                     config_.supportedFormats.end(),
                                     detectedFormat) != config_.supportedFormats.end();

    if (!formatSupported) {
        Logger::warn("Unsupported audio format detected: {}", formatToString(detectedFormat));
        return false;
    }

    return true;
}

std::string AudioUploadEndpoint::generateFileId() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);

    std::stringstream ss;
    ss << "upload_";
    for (int i = 0; i < 16; ++i) {
        ss << std::hex << dis(gen);
    }
    return ss.str();
}

std::string AudioUploadEndpoint::generateTempFilePath(const std::string& fileId, const std::string& extension) {
    return config_.tempDirectory + "/" + fileId + extension;
}

bool AudioUploadEndpoint::saveUploadedFile(const UploadRequest& request, const std::string& filePath) {
    try {
        std::ofstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            Logger::error("Failed to create temp file: {}", filePath);
            return false;
        }

        file.write(reinterpret_cast<const char*>(request.fileData.data()), request.fileData.size());
        file.close();

        Logger::info("Saved uploaded file: {} ({} bytes)", filePath, request.fileData.size());
        return true;

    } catch (const std::exception& e) {
        Logger::error("Error saving uploaded file: {}", e.what());
        return false;
    }
}

void AudioUploadEndpoint::startBackgroundProcessing(const std::string& fileId, const std::string& filePath,
                                                   const UploadRequest& request) {
    auto task = std::make_unique<ProcessingTask>();
    task->fileId = fileId;
    task->filePath = filePath;
    task->request = request;
    task->startTime = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        processingQueue_.push(std::move(task));
    }
    queueCondition_.notify_one();
}

void AudioUploadEndpoint::processingThread() {
    Logger::info("Audio upload processing thread started");

    while (!shouldShutdown_.load()) {
        std::unique_ptr<ProcessingTask> task;

        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            queueCondition_.wait(lock, [this] {
                return shouldShutdown_.load() || !processingQueue_.empty();
            });

            if (shouldShutdown_.load()) {
                break;
            }

            if (!processingQueue_.empty()) {
                task = std::move(processingQueue_.front());
                processingQueue_.pop();
            }
        }

        if (task) {
            try {
                processAudioFile(task->fileId, task->filePath, task->request);
            } catch (const std::exception& e) {
                Logger::error("Error processing audio file {}: {}", task->fileId, e.what());

                UploadResponse errorResponse;
                errorResponse.fileId = task->fileId;
                errorResponse.success = false;
                errorResponse.errorMessage = e.what();

                if (uploadErrorCallback_) {
                    uploadErrorCallback_(task->fileId, e.what());
                }

                completeProcessing(task->fileId, errorResponse);
            }
        }
    }

    Logger::info("Audio upload processing thread stopped");
}

UploadResponse AudioUploadEndpoint::processAudioFile(const std::string& fileId, const std::string& filePath,
                                                       const UploadRequest& request) {
    Logger::info("Processing audio file: {}", fileId);

    UploadResponse response;
    response.fileId = fileId;
    response.filename = request.filename;
    response.success = false;

    try {
        // Update progress
        updateProcessingProgress(fileId, ProcessingStatus::LOADING, 10.0f, "Loading file");

        // Validate file format
        if (!validateAudioFormat(filePath)) {
            response.errorMessage = "Unsupported or corrupted audio format";
            return response;
        }

        updateProcessingProgress(fileId, ProcessingStatus::LOADING, 30.0f, "Validating format");

        // Load audio file
        auto audioLoader = std::make_unique<AudioFileLoader>();
        if (!audioLoader->load(filePath)) {
            response.errorMessage = "Failed to load audio file";
            return response;
        }

        updateProcessingProgress(fileId, ProcessingStatus::PROCESSING, 50.0f, "Extracting metadata");

        // Extract metadata
        AudioMetadata metadata = audioLoader->getMetadata();
        response.extractedMetadata = metadata;
        response.detectedFormat = audioLoader->getFormat();

        // Generate preview if requested
        if (request.generatePreviews) {
            updateProcessingProgress(fileId, ProcessingStatus::PROCESSING, 70.0f, "Generating preview");
            // TODO: Generate audio preview
        }

        updateProcessingProgress(fileId, ProcessingStatus::COMPLETED, 100.0f, "Processing complete");

        response.success = true;
        response.status = ProcessingStatus::COMPLETED;

        // Update file info
        {
            std::lock_guard<std::mutex> lock(filesMutex_);
            auto it = uploadedFiles_.find(fileId);
            if (it != uploadedFiles_.end()) {
                it->second.format = response.detectedFormat;
                it->second.metadata = metadata;
                it->second.processingStatus = ProcessingStatus::COMPLETED;
                it->second.isProcessing = false;
            }
        }

        // Update statistics
        {
            std::lock_guard<std::mutex> lock(statsMutex_);
            statistics_.successfulUploads++;
            statistics_.formatCounts[response.detectedFormat]++;
        }

        Logger::info("Successfully processed audio file: {}", fileId);

    } catch (const std::exception& e) {
        Logger::error("Error processing audio file {}: {}", fileId, e.what());
        response.errorMessage = e.what();
        response.status = ProcessingStatus::ERROR;

        // Update file info
        {
            std::lock_guard<std::mutex> lock(filesMutex_);
            auto it = uploadedFiles_.find(fileId);
            if (it != uploadedFiles_.end()) {
                it->second.processingStatus = ProcessingStatus::ERROR;
                it->second.isProcessing = false;
            }
        }

        // Update statistics
        {
            std::lock_guard<std::mutex> lock(statsMutex_);
            statistics_.failedUploads++;
        }
    }

    completeProcessing(fileId, response);
    return response;
}

void AudioUploadEndpoint::updateProcessingProgress(const std::string& fileId, ProcessingStatus status,
                                                   float progress, const std::string& stage) {
    // Update progress info
    {
        std::lock_guard<std::mutex> lock(progressMutex_);
        auto it = uploadProgress_.find(fileId);
        if (it != uploadProgress_.end()) {
            it->second->status.store(status);
            it->second->progress.store(progress);
            it->second->currentStage = stage;
            it->second->stageProgress.store(progress);
            it->second->lastUpdate = std::chrono::steady_clock::now();
        }
    }

    // Update file info
    {
        std::lock_guard<std::mutex> lock(filesMutex_);
        auto it = uploadedFiles_.find(fileId);
        if (it != uploadedFiles_.end()) {
            it->second.processingStatus = status;
        }
    }

    // Trigger progress callback
    if (uploadProgressCallback_) {
        UploadProgress uploadProgress;
        uploadProgress.fileId = fileId;
        uploadProgress.status = status;
        uploadProgress.progress = progress;
        uploadProgress.currentStage = stage;
        uploadProgress.stageProgress = progress;

        {
            std::lock_guard<std::mutex> lock(progressMutex_);
            auto it = uploadProgress_.find(fileId);
            if (it != uploadProgress_.end()) {
                uploadProgress.bytesProcessed = it->second->bytesProcessed.load();
                uploadProgress.totalBytes = it->second->totalBytes.load();
            }
        }

        uploadProgressCallback_(uploadProgress);
    }
}

void AudioUploadEndpoint::completeProcessing(const std::string& fileId, const UploadResponse& response) {
    // Update final progress
    updateProcessingProgress(fileId, response.status, response.success ? 100.0f : 0.0f, "Complete");

    // Calculate final statistics
    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        auto now = std::chrono::steady_clock::now();
        auto it = uploadedFiles_.find(fileId);
        if (it != uploadedFiles_.end()) {
            auto uploadDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - it->second.uploadTime).count();

            // Update average upload time
            uint64_t totalUploads = statistics_.totalUploads;
            if (totalUploads > 1) {
                statistics_.averageUploadTime =
                    (statistics_.averageUploadTime * (totalUploads - 1) + uploadDuration) / totalUploads;
            } else {
                statistics_.averageUploadTime = uploadDuration;
            }
        }
    }

    // Trigger completion callback
    if (uploadCompleteCallback_) {
        uploadCompleteCallback_(response);
    }
}

HTTPResponse AudioUploadEndpoint::createSuccessResponse(const UploadResponse& uploadResponse) {
    HTTPResponse response;
    response.status = HttpStatus::OK;
    response.contentType = "application/json";

    std::stringstream json;
    json << "{"
         << "\"success\":true,"
         << "\"fileId\":\"" << uploadResponse.fileId << "\","
         << "\"filename\":\"" << uploadResponse.filename << "\","
         << "\"detectedFormat\":\"" << formatToString(uploadResponse.detectedFormat) << "\","
         << "\"status\":\"" << statusToString(uploadResponse.status) << "\","
         << "\"progress\":" << uploadResponse.progress << ","
         << "\"fileSize\":" << uploadResponse.fileSize << ","
         << "\"estimatedProcessingTime\":" << uploadResponse.estimatedProcessingTime;

    if (!uploadResponse.extractedMetadata.title.empty()) {
        json << ",\"metadata\":{"
             << "\"title\":\"" << uploadResponse.extractedMetadata.title << "\","
             << "\"artist\":\"" << uploadResponse.extractedMetadata.artist << "\","
             << "\"album\":\"" << uploadResponse.extractedMetadata.album << "\","
             << "\"duration\":" << uploadResponse.extractedMetadata.duration.count()
             << "}";
    }

    json << "}";

    response.body = json.str();
    return response;
}

HTTPResponse AudioUploadEndpoint::createErrorResponse(const std::string& message, int statusCode) {
    HTTPResponse response;
    response.status = static_cast<HttpStatus>(statusCode);
    response.contentType = "application/json";

    std::stringstream json;
    json << "{"
         << "\"success\":false,"
         << "\"error\":\"" << message << "\","
         << "\"statusCode\":" << statusCode
         << "}";

    response.body = json.str();
    return response;
}

bool AudioUploadEndpoint::validateAudioFormat(const std::string& filePath) {
    try {
        auto detector = std::make_unique<FormatDetector>();
        AudioFormat format = detector->detectFormat(filePath);
        return format != AudioFormat::UNKNOWN;
    } catch (const std::exception& e) {
        Logger::error("Error validating audio format: {}", e.what());
        return false;
    }
}

std::string AudioUploadEndpoint::generateTempFilePath(const std::string& fileId, const std::string& extension) {
    return config_.tempDirectory + "/" + fileId + extension;
}

// Utility implementations
std::string AudioUploadEndpoint::formatFileSize(uint64_t bytes) const {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unitIndex = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unitIndex < 4) {
        size /= 1024.0;
        unitIndex++;
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << size << " " << units[unitIndex];
    return ss.str();
}

std::string AudioUploadEndpoint::formatDuration(double seconds) const {
    int hours = static_cast<int>(seconds) / 3600;
    int minutes = (static_cast<int>(seconds) % 3600) / 60;
    int secs = static_cast<int>(seconds) % 60;

    std::stringstream ss;
    if (hours > 0) {
        ss << hours << ":";
        ss << std::setw(2) << std::setfill('0') << minutes << ":";
        ss << std::setw(2) << std::setfill('0') << secs;
    } else if (minutes > 0) {
        ss << minutes << ":";
        ss << std::setw(2) << std::setfill('0') << secs;
    } else {
        ss << secs << "s";
    }

    return ss.str();
}

bool AudioUploadEndpoint::isValidFilename(const std::string& filename) const {
    if (filename.empty() || filename.length() > 255) {
        return false;
    }

    // Check for invalid characters
    static const std::string invalidChars = "<>:\"|?*\\/";
    return filename.find_first_of(invalidChars) == std::string::npos;
}

std::string AudioUploadEndpoint::sanitizeExtension(const std::string& filename) {
    size_t dotPos = filename.find_last_of('.');
    if (dotPos != std::string::npos) {
        return filename.substr(dotPos);
    }
    return "";
}

// AudioUploadUtils implementations
AudioFormat AudioUploadUtils::detectAudioFormat(const std::vector<uint8_t>& fileData) {
    // Check file signatures for different audio formats
    if (fileData.size() < 12) {
        return AudioFormat::UNKNOWN;
    }

    // WAV format check
    if (fileData.size() >= 12 &&
        std::string(fileData.begin(), fileData.begin() + 4) == "RIFF" &&
        std::string(fileData.begin() + 8, fileData.begin() + 12) == "WAVE") {
        return AudioFormat::WAV;
    }

    // FLAC format check
    if (fileData.size() >= 4 &&
        std::string(fileData.begin(), fileData.begin() + 4) == "fLaC") {
        return AudioFormat::FLAC;
    }

    // MP3 format check (ID3v2 or sync bits)
    if (fileData.size() >= 3 && fileData[0] == 'I' && fileData[1] == 'D' && fileData[2] == '3') {
        return AudioFormat::MP3;
    }

    if (fileData.size() >= 2 && (fileData[0] == 0xFF && (fileData[1] & 0xE0) == 0xE0)) {
        return AudioFormat::MP3;
    }

    // OGG format check
    if (fileData.size() >= 4 &&
        std::string(fileData.begin(), fileData.begin() + 4) == "OggS") {
        return AudioFormat::OGG;
    }

    // DSD formats would have their own signatures
    // For now, return UNKNOWN for formats not detected

    return AudioFormat::UNKNOWN;
}

std::vector<uint8_t> AudioUploadUtils::base64Decode(const std::string& base64) {
    // Simple base64 decoding implementation
    static const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<uint8_t> result;

    int val = 0, valb = -8;
    for (char c : base64) {
        if (c == '=') break;
        if (c == '-' || c == '_' || c == '\n' || c == '\r') continue;

        auto pos = chars.find(c);
        if (pos == std::string::npos) continue;

        val = (val << 6) + pos;
        valb += 6;
        if (valb >= 0) {
            result.push_back((val >> 8) & 0xff);
            valb -= 8;
        }
    }

    return result;
}

} // namespace vortex