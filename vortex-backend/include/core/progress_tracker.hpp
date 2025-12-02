#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <functional>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include "system/logger.hpp"

namespace vortex::core {

/**
 * Progress tracking stages for audio processing
 */
enum class ProcessingStage {
    Unknown,
    Initializing,
    LoadingFile,
    Decoding,
    Analyzing,
    Processing,
    ApplyingEffects,
    Encoding,
    Saving,
    Completed,
    Error
};

/**
 * Processing task information
 */
struct ProcessingTask {
    std::string id;
    std::string name;
    std::string filePath;
    ProcessingStage stage;
    float progress;              // 0.0 to 1.0
    std::string message;         // Human-readable status message
    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point estimatedEndTime;
    uint64_t bytesProcessed;
    uint64_t totalBytes;
    uint32_t currentStep;
    uint32_t totalSteps;
    std::string error;
    bool cancellable;
    bool cancelled;

    ProcessingTask() : stage(ProcessingStage::Unknown), progress(0.0f),
                       bytesProcessed(0), totalBytes(0), currentStep(0),
                       totalSteps(1), cancellable(true), cancelled(false) {}
};

/**
 * Progress callback function type
 */
using ProgressCallback = std::function<void(const ProcessingTask&)>;

/**
 * Processing progress tracker
 *
 * This class tracks the progress of audio processing operations,
 * provides callbacks for progress updates, and supports concurrent
 * processing of multiple tasks with cancellation capabilities.
 */
class ProgressTracker {
public:
    /**
     * Constructor
     */
    ProgressTracker();

    /**
     * Destructor
     */
    ~ProgressTracker();

    /**
     * Create a new processing task
     * @param name Task name
     * @param filePath Path to file being processed (optional)
     * @param cancellable Whether the task can be cancelled
     * @return Unique task ID
     */
    std::string createTask(const std::string& name,
                          const std::string& filePath = "",
                          bool cancellable = true);

    /**
     * Update task progress
     * @param taskId Task ID
     * @param progress Progress value (0.0 to 1.0)
     * @param message Status message (optional)
     */
    void updateProgress(const std::string& taskId,
                       float progress,
                       const std::string& message = "");

    /**
     * Update task stage
     * @param taskId Task ID
     * @param stage New processing stage
     * @param message Status message (optional)
     */
    void updateStage(const std::string& taskId,
                    ProcessingStage stage,
                    const std::string& message = "");

    /**
     * Update task progress with step information
     * @param taskId Task ID
     * @param currentStep Current step number
     * @param totalSteps Total number of steps
     * @param message Status message (optional)
     */
    void updateStepProgress(const std::string& taskId,
                           uint32_t currentStep,
                           uint32_t totalSteps,
                           const std::string& message = "");

    /**
     * Update task progress with byte information
     * @param taskId Task ID
     * @param bytesProcessed Bytes processed so far
     * @param totalBytes Total bytes to process
     * @param message Status message (optional)
     */
    void updateByteProgress(const std::string& taskId,
                           uint64_t bytesProcessed,
                           uint64_t totalBytes,
                           const std::string& message = "");

    /**
     * Mark task as completed
     * @param taskId Task ID
     * @param message Completion message (optional)
     */
    void completeTask(const std::string& taskId,
                     const std::string& message = "Task completed successfully");

    /**
     * Mark task as failed
     * @param taskId Task ID
     * @param error Error message
     */
    void failTask(const std::string& taskId, const std::string& error);

    /**
     * Cancel a task
     * @param taskId Task ID
     * @return true if task was cancelled, false if not cancellable or not found
     */
    bool cancelTask(const std::string& taskId);

    /**
     * Check if a task is cancelled
     * @param taskId Task ID
     * @return true if task is cancelled
     */
    bool isTaskCancelled(const std::string& taskId) const;

    /**
     * Get task information
     * @param taskId Task ID
     * @return Task information (nullptr if not found)
     */
    std::shared_ptr<ProcessingTask> getTask(const std::string& taskId) const;

    /**
     * Get all active tasks
     * @return Vector of active task information
     */
    std::vector<ProcessingTask> getActiveTasks() const;

    /**
     * Get completed tasks
     * @param limit Maximum number of tasks to return (0 = no limit)
     * @return Vector of completed task information
     */
    std::vector<ProcessingTask> getCompletedTasks(size_t limit = 0) const;

    /**
     * Remove a task from tracking
     * @param taskId Task ID
     */
    void removeTask(const std::string& taskId);

    /**
     * Clear all completed tasks
     */
    void clearCompletedTasks();

    /**
     * Set progress callback
     * @param callback Function to call when progress updates
     */
    void setProgressCallback(ProgressCallback callback);

    /**
     * Set completion callback
     * @param callback Function to call when a task completes
     */
    void setCompletionCallback(ProgressCallback callback);

    /**
     * Set error callback
     * @param callback Function to call when a task fails
     */
    void setErrorCallback(ProgressCallback callback);

    /**
     * Get processing statistics
     * @return JSON string with statistics
     */
    std::string getStatistics() const;

    /**
     * Get stage name as string
     * @param stage Processing stage
     * @return Stage name
     */
    static std::string getStageName(ProcessingStage stage);

    /**
     * Estimate remaining time for a task
     * @param taskId Task ID
     * @return Estimated remaining time in seconds, or -1 if unknown
     */
    double getEstimatedRemainingTime(const std::string& taskId) const;

    /**
     * Create a scoped progress guard for automatic cleanup
     * @param taskId Task ID
     * @param finalStage Final stage to set when guard goes out of scope
     * @return Progress guard object
     */
    std::unique_ptr<ProgressGuard> createGuard(const std::string& taskId,
                                             ProcessingStage finalStage = ProcessingStage::Completed);

    /**
     * Progress guard class for automatic task management
     */
    class ProgressGuard {
    public:
        ProgressGuard(ProgressTracker& tracker, const std::string& taskId, ProcessingStage finalStage);
        ~ProgressGuard();

    private:
        ProgressTracker& tracker_;
        std::string taskId_;
        ProcessingStage finalStage_;
        bool completed_;
    };

private:
    mutable std::mutex tasksMutex_;
    mutable std::mutex callbacksMutex_;

    std::map<std::string, std::shared_ptr<ProcessingTask>> activeTasks_;
    std::vector<ProcessingTask> completedTasks_;
    std::atomic<size_t> completedTaskCount_;

    ProgressCallback progressCallback_;
    ProgressCallback completionCallback_;
    ProgressCallback errorCallback_;

    // Background cleanup thread
    std::thread cleanupThread_;
    std::atomic<bool> shutdownRequested_;
    std::condition_variable cleanupCondition_;
    mutable std::mutex cleanupMutex_;

    /**
     * Notify callbacks for task update
     * @param task Updated task
     * @param isCompletion Whether this is a completion event
     * @param isError Whether this is an error event
     */
    void notifyCallbacks(const ProcessingTask& task, bool isCompletion, bool isError);

    /**
     * Cleanup old completed tasks
     */
    void cleanupOldTasks();

    /**
     * Background cleanup thread function
     */
    void cleanupThreadFunction();

    /**
     * Update task internal state and calculate derived values
     * @param task Task to update
     */
    void updateTaskDerivedState(ProcessingTask& task);
};

/**
 * RAII progress guard utility
 *
 * This class automatically manages task lifecycle by setting
 * a final stage when the guard goes out of scope.
 */
class ScopedProgressTracker {
public:
    ScopedProgressTracker(ProgressTracker& tracker,
                         const std::string& name,
                         const std::string& filePath = "",
                         ProcessingStage finalStage = ProcessingStage::Completed);

    ~ScopedProgressTracker();

    /**
     * Get the task ID
     */
    const std::string& getTaskId() const { return taskId_; }

    /**
     * Get the progress tracker
     */
    ProgressTracker& getTracker() { return tracker_; }

    /**
     * Update progress
     */
    void updateProgress(float progress, const std::string& message = "");

    /**
     * Update stage
     */
    void updateStage(ProcessingStage stage, const std::string& message = "");

    /**
     * Update step progress
     */
    void updateStepProgress(uint32_t currentStep, uint32_t totalSteps, const std::string& message = "");

    /**
     * Update byte progress
     */
    void updateByteProgress(uint64_t bytesProcessed, uint64_t totalBytes, const std::string& message = "");

    /**
     * Complete the task early
     */
    void complete(const std::string& message = "Task completed");

    /**
     * Fail the task
     */
    void fail(const std::string& error);

private:
    ProgressTracker& tracker_;
    std::string taskId_;
    ProcessingStage finalStage_;
    bool completed_;
};

} // namespace vortex::core