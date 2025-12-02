#include "core/progress_tracker.hpp"
#include <algorithm>
#include <random>
#include <iomanip>
#include <sstream>

namespace vortex::core {

ProgressTracker::ProgressTracker()
    : completedTaskCount_(0)
    , shutdownRequested_(false) {
    // Start background cleanup thread
    cleanupThread_ = std::thread(&ProgressTracker::cleanupThreadFunction, this);
    Logger::info("ProgressTracker initialized");
}

ProgressTracker::~ProgressTracker() {
    // Signal shutdown to cleanup thread
    shutdownRequested_.store(true);
    cleanupCondition_.notify_all();

    // Wait for cleanup thread to finish
    if (cleanupThread_.joinable()) {
        cleanupThread_.join();
    }

    Logger::info("ProgressTracker shutdown");
}

std::string ProgressTracker::createTask(const std::string& name,
                                       const std::string& filePath,
                                       bool cancellable) {
    // Generate unique task ID
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 35); // 0-9 and A-Z
    std::string taskId;
    for (int i = 0; i < 16; ++i) {
        int value = dis(gen);
        taskId += (value < 10) ? std::to_string(value) : char('A' + (value - 10));
    }

    std::lock_guard<std::mutex> lock(tasksMutex_);

    auto task = std::make_shared<ProcessingTask>();
    task->id = taskId;
    task->name = name;
    task->filePath = filePath;
    task->stage = ProcessingStage::Initializing;
    task->progress = 0.0f;
    task->message = "Initializing task...";
    task->startTime = std::chrono::steady_clock::now();
    task->bytesProcessed = 0;
    task->totalBytes = 0;
    task->currentStep = 0;
    task->totalSteps = 1;
    task->cancellable = cancellable;
    task->cancelled = false;

    activeTasks_[taskId] = task;

    Logger::debug("Created task: {} ({})", taskId, name);
    notifyCallbacks(*task, false, false);

    return taskId;
}

void ProgressTracker::updateProgress(const std::string& taskId,
                                    float progress,
                                    const std::string& message) {
    std::lock_guard<std::mutex> lock(tasksMutex_);

    auto it = activeTasks_.find(taskId);
    if (it != activeTasks_.end()) {
        auto& task = *it->second;
        task.progress = std::clamp(progress, 0.0f, 1.0f);
        if (!message.empty()) {
            task.message = message;
        }

        updateTaskDerivedState(task);
        notifyCallbacks(task, false, false);
    }
}

void ProgressTracker::updateStage(const std::string& taskId,
                                 ProcessingStage stage,
                                 const std::string& message) {
    std::lock_guard<std::mutex> lock(tasksMutex_);

    auto it = activeTasks_.find(taskId);
    if (it != activeTasks_.end()) {
        auto& task = *it->second;
        task.stage = stage;
        if (!message.empty()) {
            task.message = message;
        } else {
            task.message = getStageName(stage);
        }

        updateTaskDerivedState(task);
        notifyCallbacks(task, false, false);
    }
}

void ProgressTracker::updateStepProgress(const std::string& taskId,
                                        uint32_t currentStep,
                                        uint32_t totalSteps,
                                        const std::string& message) {
    std::lock_guard<std::mutex> lock(tasksMutex_);

    auto it = activeTasks_.find(taskId);
    if (it != activeTasks_.end()) {
        auto& task = *it->second;
        task.currentStep = currentStep;
        task.totalSteps = totalSteps;

        if (totalSteps > 0) {
            task.progress = static_cast<float>(currentStep) / static_cast<float>(totalSteps);
        }

        if (!message.empty()) {
            task.message = message;
        }

        updateTaskDerivedState(task);
        notifyCallbacks(task, false, false);
    }
}

void ProgressTracker::updateByteProgress(const std::string& taskId,
                                        uint64_t bytesProcessed,
                                        uint64_t totalBytes,
                                        const std::string& message) {
    std::lock_guard<std::mutex> lock(tasksMutex_);

    auto it = activeTasks_.find(taskId);
    if (it != activeTasks_.end()) {
        auto& task = *it->second;
        task.bytesProcessed = bytesProcessed;
        task.totalBytes = totalBytes;

        if (totalBytes > 0) {
            task.progress = static_cast<float>(bytesProcessed) / static_cast<float>(totalBytes);
        }

        if (!message.empty()) {
            task.message = message;
        }

        updateTaskDerivedState(task);
        notifyCallbacks(task, false, false);
    }
}

void ProgressTracker::completeTask(const std::string& taskId, const std::string& message) {
    std::lock_guard<std::mutex> lock(tasksMutex_);

    auto it = activeTasks_.find(taskId);
    if (it != activeTasks_.end()) {
        auto& task = *it->second;
        task.stage = ProcessingStage::Completed;
        task.progress = 1.0f;
        task.message = message.empty() ? "Task completed successfully" : message;
        task.currentStep = task.totalSteps;
        task.bytesProcessed = task.totalBytes;

        updateTaskDerivedState(task);

        // Move to completed tasks
        completedTasks_.push_back(task);
        completedTaskCount_++;
        activeTasks_.erase(it);

        notifyCallbacks(task, true, false);
        Logger::info("Task completed: {} ({})", taskId, task.name);
    }
}

void ProgressTracker::failTask(const std::string& taskId, const std::string& error) {
    std::lock_guard<std::mutex> lock(tasksMutex_);

    auto it = activeTasks_.find(taskId);
    if (it != activeTasks_.end()) {
        auto& task = *it->second;
        task.stage = ProcessingStage::Error;
        task.error = error;
        task.message = "Error: " + error;

        updateTaskDerivedState(task);

        // Move to completed tasks
        completedTasks_.push_back(task);
        completedTaskCount_++;
        activeTasks_.erase(it);

        notifyCallbacks(task, false, true);
        Logger::error("Task failed: {} ({}) - {}", taskId, task.name, error);
    }
}

bool ProgressTracker::cancelTask(const std::string& taskId) {
    std::lock_guard<std::mutex> lock(tasksMutex_);

    auto it = activeTasks_.find(taskId);
    if (it != activeTasks_.end()) {
        auto& task = *it->second;
        if (task.cancellable) {
            task.cancelled = true;
            task.message = "Task cancelled";
            task.stage = ProcessingStage::Error;
            task.error = "Task was cancelled by user";

            updateTaskDerivedState(task);

            // Move to completed tasks
            completedTasks_.push_back(task);
            completedTaskCount_++;
            activeTasks_.erase(it);

            notifyCallbacks(task, false, true);
            Logger::info("Task cancelled: {} ({})", taskId, task.name);
            return true;
        }
    }

    return false;
}

bool ProgressTracker::isTaskCancelled(const std::string& taskId) const {
    std::lock_guard<std::mutex> lock(tasksMutex_);

    auto it = activeTasks_.find(taskId);
    if (it != activeTasks_.end()) {
        return it->second->cancelled;
    }

    return false;
}

std::shared_ptr<ProcessingTask> ProgressTracker::getTask(const std::string& taskId) const {
    std::lock_guard<std::mutex> lock(tasksMutex_);

    auto it = activeTasks_.find(taskId);
    if (it != activeTasks_.end()) {
        return it->second;
    }

    // Check completed tasks
    for (const auto& task : completedTasks_) {
        if (task.id == taskId) {
            return std::make_shared<ProcessingTask>(task);
        }
    }

    return nullptr;
}

std::vector<ProcessingTask> ProgressTracker::getActiveTasks() const {
    std::lock_guard<std::mutex> lock(tasksMutex_);

    std::vector<ProcessingTask> tasks;
    tasks.reserve(activeTasks_.size());

    for (const auto& pair : activeTasks_) {
        tasks.push_back(*pair.second);
    }

    return tasks;
}

std::vector<ProcessingTask> ProgressTracker::getCompletedTasks(size_t limit) const {
    std::lock_guard<std::mutex> lock(tasksMutex_);

    std::vector<ProcessingTask> tasks;

    if (limit == 0) {
        tasks = completedTasks_;
    } else {
        // Return the most recent completed tasks
        size_t startIdx = (completedTasks_.size() > limit) ? completedTasks_.size() - limit : 0;
        for (size_t i = startIdx; i < completedTasks_.size(); ++i) {
            tasks.push_back(completedTasks_[i]);
        }
    }

    return tasks;
}

void ProgressTracker::removeTask(const std::string& taskId) {
    std::lock_guard<std::mutex> lock(tasksMutex_);

    activeTasks_.erase(taskId);

    // Remove from completed tasks as well
    completedTasks_.erase(
        std::remove_if(completedTasks_.begin(), completedTasks_.end(),
                     [&taskId](const ProcessingTask& task) { return task.id == taskId; }),
        completedTasks_.end());
}

void ProgressTracker::clearCompletedTasks() {
    std::lock_guard<std::mutex> lock(tasksMutex_);
    completedTasks_.clear();
    completedTaskCount_.store(0);
}

void ProgressTracker::setProgressCallback(ProgressCallback callback) {
    std::lock_guard<std::mutex> lock(callbacksMutex_);
    progressCallback_ = callback;
}

void ProgressTracker::setCompletionCallback(ProgressCallback callback) {
    std::lock_guard<std::mutex> lock(callbacksMutex_);
    completionCallback_ = callback;
}

void ProgressTracker::setErrorCallback(ProgressCallback callback) {
    std::lock_guard<std::mutex> lock(callbacksMutex_);
    errorCallback_ = callback;
}

std::string ProgressTracker::getStatistics() const {
    std::lock_guard<std::mutex> lock(tasksMutex_);

    size_t activeCount = activeTasks_.size();
    size_t completedCount = completedTasks_.size();

    // Calculate average progress for active tasks
    float totalProgress = 0.0f;
    for (const auto& pair : activeTasks_) {
        totalProgress += pair.second->progress;
    }
    float avgProgress = activeCount > 0 ? totalProgress / static_cast<float>(activeCount) : 0.0f;

    // Count tasks by stage
    std::map<ProcessingStage, size_t> stageCounts;
    for (const auto& pair : activeTasks_) {
        stageCounts[pair.second->stage]++;
    }

    // Create JSON output
    std::ostringstream oss;
    oss << "{"
        << "\"active_tasks\":" << activeCount << ","
        << "\"completed_tasks\":" << completedCount << ","
        << "\"total_completed\":" << completedTaskCount_.load() << ","
        << "\"average_progress\":" << std::fixed << std::setprecision(3) << avgProgress << ","
        << "\"stages\":{";

    bool firstStage = true;
    for (const auto& pair : stageCounts) {
        if (!firstStage) oss << ",";
        oss << "\"" << getStageName(pair.first) << "\":" << pair.second;
        firstStage = false;
    }

    oss << "}}";

    return oss.str();
}

std::string ProgressTracker::getStageName(ProcessingStage stage) {
    switch (stage) {
        case ProcessingStage::Initializing: return "Initializing";
        case ProcessingStage::LoadingFile: return "LoadingFile";
        case ProcessingStage::Decoding: return "Decoding";
        case ProcessingStage::Analyzing: return "Analyzing";
        case ProcessingStage::Processing: return "Processing";
        case ProcessingStage::ApplyingEffects: return "ApplyingEffects";
        case ProcessingStage::Encoding: return "Encoding";
        case ProcessingStage::Saving: return "Saving";
        case ProcessingStage::Completed: return "Completed";
        case ProcessingStage::Error: return "Error";
        default: return "Unknown";
    }
}

double ProgressTracker::getEstimatedRemainingTime(const std::string& taskId) const {
    std::lock_guard<std::mutex> lock(tasksMutex_);

    auto it = activeTasks_.find(taskId);
    if (it == activeTasks_.end()) {
        return -1.0;
    }

    const auto& task = *it->second;
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - task.startTime);

    if (task.progress > 0.01f && elapsed.count() > 0) {
        double estimatedTotal = static_cast<double>(elapsed.count()) / static_cast<double>(task.progress);
        double remaining = estimatedTotal - static_cast<double>(elapsed.count());
        return std::max(0.0, remaining);
    }

    return -1.0;
}

std::unique_ptr<ProgressTracker::ProgressGuard> ProgressTracker::createGuard(
    const std::string& taskId, ProcessingStage finalStage) {
    return std::make_unique<ProgressGuard>(*this, taskId, finalStage);
}

void ProgressTracker::notifyCallbacks(const ProcessingTask& task, bool isCompletion, bool isError) {
    std::lock_guard<std::mutex> lock(callbacksMutex_);

    if (progressCallback_ && !isCompletion && !isError) {
        try {
            progressCallback_(task);
        } catch (const std::exception& e) {
            Logger::error("Progress callback exception: {}", e.what());
        }
    }

    if (completionCallback_ && isCompletion && !isError) {
        try {
            completionCallback_(task);
        } catch (const std::exception& e) {
            Logger::error("Completion callback exception: {}", e.what());
        }
    }

    if (errorCallback_ && isError) {
        try {
            errorCallback_(task);
        } catch (const std::exception& e) {
            Logger::error("Error callback exception: {}", e.what());
        }
    }
}

void ProgressTracker::cleanupOldTasks() {
    std::lock_guard<std::mutex> lock(tasksMutex_);

    // Keep only the most recent 100 completed tasks
    const size_t maxCompletedTasks = 100;
    if (completedTasks_.size() > maxCompletedTasks) {
        size_t toRemove = completedTasks_.size() - maxCompletedTasks;
        completedTasks_.erase(completedTasks_.begin(), completedTasks_.begin() + toRemove);
    }
}

void ProgressTracker::cleanupThreadFunction() {
    auto interval = std::chrono::minutes(5); // Cleanup every 5 minutes

    while (!shutdownRequested_.load()) {
        std::unique_lock<std::mutex> lock(cleanupMutex_);

        // Wait for interval or shutdown signal
        cleanupCondition_.wait_for(lock, interval, [this] { return shutdownRequested_.load(); });

        if (!shutdownRequested_.load()) {
            cleanupOldTasks();
        }
    }
}

void ProgressTracker::updateTaskDerivedState(ProcessingTask& task) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - task.startTime);

    if (task.progress > 0.01f && elapsed.count() > 0) {
        double estimatedTotal = static_cast<double>(elapsed.count()) / static_cast<double>(task.progress);
        task.estimatedEndTime = task.startTime + std::chrono::seconds(static_cast<int64_t>(estimatedTotal));
    }
}

// ProgressGuard implementation
ProgressTracker::ProgressGuard::ProgressGuard(ProgressTracker& tracker,
                                                const std::string& taskId,
                                                ProcessingStage finalStage)
    : tracker_(tracker)
    , taskId_(taskId)
    , finalStage_(finalStage)
    , completed_(false) {
}

ProgressTracker::ProgressGuard::~ProgressGuard() {
    if (!completed_) {
        auto task = tracker_.getTask(taskId_);
        if (task && task->stage != ProcessingStage::Error) {
            if (finalStage_ == ProcessingStage::Completed) {
                tracker_.completeTask(taskId_);
            } else {
                tracker_.updateStage(taskId_, finalStage_);
            }
        }
    }
}

// ScopedProgressTracker implementation
ScopedProgressTracker::ScopedProgressTracker(ProgressTracker& tracker,
                                           const std::string& name,
                                           const std::string& filePath,
                                           ProcessingStage finalStage)
    : tracker_(tracker)
    , finalStage_(finalStage)
    , completed_(false) {
    taskId_ = tracker_.createTask(name, filePath);
}

ScopedProgressTracker::~ScopedProgressTracker() {
    if (!completed_) {
        if (finalStage_ == ProcessingStage::Completed) {
            tracker_.completeTask(taskId_);
        } else {
            tracker_.updateStage(taskId_, finalStage_);
        }
    }
}

void ScopedProgressTracker::updateProgress(float progress, const std::string& message) {
    tracker_.updateProgress(taskId_, progress, message);
}

void ScopedProgressTracker::updateStage(ProcessingStage stage, const std::string& message) {
    tracker_.updateStage(taskId_, stage, message);
}

void ScopedProgressTracker::updateStepProgress(uint32_t currentStep, uint32_t totalSteps, const std::string& message) {
    tracker_.updateStepProgress(taskId_, currentStep, totalSteps, message);
}

void ScopedProgressTracker::updateByteProgress(uint64_t bytesProcessed, uint64_t totalBytes, const std::string& message) {
    tracker_.updateByteProgress(taskId_, bytesProcessed, totalBytes, message);
}

void ScopedProgressTracker::complete(const std::string& message) {
    completed_ = true;
    tracker_.completeTask(taskId_, message);
}

void ScopedProgressTracker::fail(const std::string& error) {
    completed_ = true;
    tracker_.failTask(taskId_, error);
}

} // namespace vortex::core