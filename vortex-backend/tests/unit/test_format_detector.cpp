#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <fstream>
#include <filesystem>
#include <vector>
#include <map>
#include "core/fileio/format_detector.hpp"
#include "system/logger.hpp"

using namespace vortex;
using namespace vortex::testing;
using namespace testing;

// Mock magic numbers for different audio formats
namespace {
    // WAV format magic numbers
    const std::vector<uint8_t> WAV_HEADER = {
        0x52, 0x49, 0x46, 0x46,  // "RIFF"
        0x57, 0x41, 0x56, 0x45   // "WAVE"
    };

    // FLAC format magic numbers
    const std::vector<uint8_t> FLAC_HEADER = {
        0x66, 0x4C, 0x61, 0x43   // "fLaC"
    };

    // MP3 format magic numbers (ID3v1 tag)
    const std::vector<uint8_t> MP3_ID3V1_HEADER = {
        0x49, 0x44, 0x33          // "ID3"
    };

    // MP3 format magic numbers (MPEG sync)
    const std::vector<uint8_t> MP3_SYNC_HEADER = {
        0xFF, 0xFB, 0x90          // MPEG sync pattern
    };

    // OGG format magic numbers
    const std::vector<uint8_t> OGG_HEADER = {
        0x4F, 0x67, 0x67, 0x53   // "OggS"
    };

    // M4A/AAC format magic numbers
    const std::vector<uint8_t> M4A_HEADER = {
        0x00, 0x00, 0x00, 0x20,  // Box size
        0x66, 0x74, 0x79, 0x70   // "ftyp"
    };

    // DFF format magic numbers (DSDIFF)
    const std::vector<uint8_t> DFF_HEADER = {
        0x46, 0x52, 0x4D, 0x38   // "FRM8"
    };

    // DSF format magic numbers
    const std::vector<uint8_t> DSF_HEADER = {
        0x44, 0x53, 0x46, 0x20   // "DSF "
    };

    // Helper function to create test audio file with magic number
    std::string createTestAudioFile(const std::string& filename,
                                   const std::vector<uint8_t>& header,
                                   size_t additionalData = 1024) {
        std::string testFile = std::filesystem::temp_directory_path().string() + "/" + filename;
        std::ofstream file(testFile, std::ios::binary);

        // Write header
        file.write(reinterpret_cast<const char*>(header.data()), header.size());

        // Write additional data
        std::vector<char> data(additionalData, 0x00);
        file.write(data.data(), data.size());

        file.close();
        return testFile;
    }

    // Helper function to create invalid audio file
    std::string createInvalidTestFile(const std::string& filename) {
        std::string testFile = std::filesystem::temp_directory_path().string() + "/" + filename;
        std::ofstream file(testFile, std::ios::binary);

        // Write invalid header
        std::vector<uint8_t> invalidHeader = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
        file.write(reinterpret_cast<const char*>(invalidHeader.data()), invalidHeader.size());

        // Write additional data
        std::vector<char> data(1024, 0x00);
        file.write(data.data(), data.size());

        file.close();
        return testFile;
    }
}

class FormatDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::initialize();
        formatDetector_ = std::make_unique<FormatDetector>();
        ASSERT_TRUE(formatDetector_->initialize());
    }

    void TearDown() override {
        // Clean up test files
        for (const auto& file : testFiles_) {
            if (std::filesystem::exists(file)) {
                std::filesystem::remove(file);
            }
        }

        formatDetector_.reset();
        Logger::shutdown();
    }

    void addTestFile(const std::string& filename, const std::vector<uint8_t>& header) {
        testFiles_.push_back(createTestAudioFile(filename, header));
    }

    void addInvalidTestFile(const std::string& filename) {
        testFiles_.push_back(createInvalidTestFile(filename));
    }

    std::unique_ptr<FormatDetector> formatDetector_;
    std::vector<std::string> testFiles_;
};

// Test WAV format detection
TEST_F(FormatDetectorTest, DetectWAVFormat) {
    // Arrange
    addTestFile("test_audio.wav", WAV_HEADER);
    std::string filePath = testFiles_.back();

    // Act
    auto result = formatDetector_->detectFormat(filePath);

    // Assert
    ASSERT_TRUE(result.has_value()) << "WAV format should be detected";
    EXPECT_EQ(result->format, AudioFormat::WAV) << "Detected format should be WAV";
    EXPECT_EQ(result->extension, "wav") << "Extension should be wav";
    EXPECT_FALSE(result->codec.empty()) << "Codec should be detected";
    EXPECT_GT(result->sampleRate, 0) << "Sample rate should be positive";
    EXPECT_GT(result->channels, 0) << "Channel count should be positive";
    EXPECT_GT(result->bitDepth, 0) << "Bit depth should be positive";
}

// Test FLAC format detection
TEST_F(FormatDetectorTest, DetectFLACFormat) {
    // Arrange
    addTestFile("test_audio.flac", FLAC_HEADER);
    std::string filePath = testFiles_.back();

    // Act
    auto result = formatDetector_->detectFormat(filePath);

    // Assert
    ASSERT_TRUE(result.has_value()) << "FLAC format should be detected";
    EXPECT_EQ(result->format, AudioFormat::FLAC) << "Detected format should be FLAC";
    EXPECT_EQ(result->extension, "flac") << "Extension should be flac";
    EXPECT_FALSE(result->codec.empty()) << "Codec should be detected";
}

// Test MP3 format detection with ID3v1 tag
TEST_F(FormatDetectorTest, DetectMP3FormatWithID3) {
    // Arrange
    addTestFile("test_audio.mp3", MP3_ID3V1_HEADER);
    std::string filePath = testFiles_.back();

    // Act
    auto result = formatDetector_->detectFormat(filePath);

    // Assert
    ASSERT_TRUE(result.has_value()) << "MP3 format with ID3 should be detected";
    EXPECT_EQ(result->format, AudioFormat::MP3) << "Detected format should be MP3";
    EXPECT_EQ(result->extension, "mp3") << "Extension should be mp3";
    EXPECT_FALSE(result->codec.empty()) << "Codec should be detected";
}

// Test MP3 format detection with sync header
TEST_F(FormatDetectorTest, DetectMP3FormatWithSync) {
    // Arrange
    addTestFile("test_audio_sync.mp3", MP3_SYNC_HEADER);
    std::string filePath = testFiles_.back();

    // Act
    auto result = formatDetector_->detectFormat(filePath);

    // Assert
    ASSERT_TRUE(result.has_value()) << "MP3 format with sync should be detected";
    EXPECT_EQ(result->format, AudioFormat::MP3) << "Detected format should be MP3";
    EXPECT_EQ(result->extension, "mp3") << "Extension should be mp3";
}

// Test OGG format detection
TEST_F(FormatDetectorTest, DetectOGGFormat) {
    // Arrange
    addTestFile("test_audio.ogg", OGG_HEADER);
    std::string filePath = testFiles_.back();

    // Act
    auto result = formatDetector_->detectFormat(filePath);

    // Assert
    ASSERT_TRUE(result.has_value()) << "OGG format should be detected";
    EXPECT_EQ(result->format, AudioFormat::OGG) << "Detected format should be OGG";
    EXPECT_EQ(result->extension, "ogg") << "Extension should be ogg";
}

// Test M4A format detection
TEST_F(FormatDetectorTest, DetectM4AFormat) {
    // Arrange
    addTestFile("test_audio.m4a", M4A_HEADER);
    std::string filePath = testFiles_.back();

    // Act
    auto result = formatDetector_->detectFormat(filePath);

    // Assert
    ASSERT_TRUE(result.has_value()) << "M4A format should be detected";
    EXPECT_EQ(result->format, AudioFormat::M4A) << "Detected format should be M4A";
    EXPECT_EQ(result->extension, "m4a") << "Extension should be m4a";
}

// Test DFF format detection (DSDIFF)
TEST_F(FormatDetectorTest, DetectDFFFormat) {
    // Arrange
    addTestFile("test_audio.dff", DFF_HEADER);
    std::string filePath = testFiles_.back();

    // Act
    auto result = formatDetector_->detectFormat(filePath);

    // Assert
    ASSERT_TRUE(result.has_value()) << "DFF format should be detected";
    EXPECT_EQ(result->format, AudioFormat::DFF) << "Detected format should be DFF";
    EXPECT_EQ(result->extension, "dff") << "Extension should be dff";
}

// Test DSF format detection
TEST_F(FormatDetectorTest, DetectDSFFormat) {
    // Arrange
    addTestFile("test_audio.dsf", DSF_HEADER);
    std::string filePath = testFiles_.back();

    // Act
    auto result = formatDetector_->detectFormat(filePath);

    // Assert
    ASSERT_TRUE(result.has_value()) << "DSF format should be detected";
    EXPECT_EQ(result->format, AudioFormat::DSF) << "Detected format should be DSF";
    EXPECT_EQ(result->extension, "dsf") << "Extension should be dsf";
}

// Test invalid format detection
TEST_F(FormatDetectorTest, DetectInvalidFormat) {
    // Arrange
    addInvalidTestFile("invalid_file.bin");
    std::string filePath = testFiles_.back();

    // Act
    auto result = formatDetector_->detectFormat(filePath);

    // Assert
    EXPECT_FALSE(result.has_value()) << "Invalid format should not be detected";
}

// Test non-existent file
TEST_F(FormatDetectorTest, DetectNonExistentFile) {
    // Arrange
    std::string nonExistentFile = "/path/to/non/existent/file.mp3";

    // Act
    auto result = formatDetector_->detectFormat(nonExistentFile);

    // Assert
    EXPECT_FALSE(result.has_value()) << "Non-existent file should not be detected";
}

// Test empty file
TEST_F(FormatDetectorTest, DetectEmptyFile) {
    // Arrange
    std::string emptyFile = std::filesystem::temp_directory_path().string() + "/empty_file.wav";
    std::ofstream file(emptyFile, std::ios::binary);
    file.close();

    // Act
    auto result = formatDetector_->detectFormat(emptyFile);

    // Assert
    EXPECT_FALSE(result.has_value()) << "Empty file should not be detected";

    // Cleanup
    std::filesystem::remove(emptyFile);
}

// Test very small file
TEST_F(FormatDetectorTest, DetectVerySmallFile) {
    // Arrange - Create file with only 4 bytes (smaller than any valid header)
    std::string smallFile = std::filesystem::temp_directory_path().string() + "/small_file.wav";
    std::ofstream file(smallFile, std::ios::binary);
    std::vector<uint8_t> tinyData = {0x01, 0x02, 0x03, 0x04};
    file.write(reinterpret_cast<const char*>(tinyData.data()), tinyData.size());
    file.close();

    // Act
    auto result = formatDetector_->detectFormat(smallFile);

    // Assert
    EXPECT_FALSE(result.has_value()) << "Very small file should not be detected as valid audio";

    // Cleanup
    std::filesystem::remove(smallFile);
}

// Test format detection from file extension fallback
TEST_F(FormatDetectorTest, DetectFormatFromExtension) {
    // Arrange - Create file with invalid header but valid extension
    std::string fileWithExtension = std::filesystem::temp_directory_path().string() + "/test.mp3";
    addInvalidTestFile("test_invalid.txt");
    std::string invalidFile = testFiles_.back();

    // Rename to have .mp3 extension
    std::filesystem::rename(invalidFile, fileWithExtension);
    testFiles_.back() = fileWithExtension;  // Update test file list

    // Act
    auto result = formatDetector_->detectFormat(fileWithExtension);

    // Assert - Should detect from extension as fallback
    ASSERT_TRUE(result.has_value()) << "Format should be detected from extension";
    EXPECT_EQ(result->format, AudioFormat::MP3) << "Should detect MP3 from extension";
    EXPECT_EQ(result->extension, "mp3") << "Extension should be mp3";
}

// Test format detection performance
TEST_F(FormatDetectorTest, FormatDetectionPerformance) {
    const int numFiles = 100;
    std::vector<std::string> testFilePaths;

    // Arrange - Create multiple test files
    for (int i = 0; i < numFiles; ++i) {
        std::string filename = "perf_test_" + std::to_string(i) + ".wav";
        addTestFile(filename, WAV_HEADER);
        testFilePaths.push_back(testFiles_.back());
    }

    // Act - Measure detection time
    auto startTime = std::chrono::high_resolution_clock::now();

    for (const auto& filePath : testFilePaths) {
        auto result = formatDetector_->detectFormat(filePath);
        EXPECT_TRUE(result.has_value()) << "Each file should be detected";
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // Assert - Should complete within reasonable time
    EXPECT_LT(duration.count(), 1000) << "Format detection should complete within 1 second for 100 files";
}

// Test DSD format detection with high sample rates
TEST_F(FormatDetectorTest, DetectDSDHighSampleRate) {
    // For DSD formats, test if detector can handle very high sample rates
    addTestFile("dsd1024_test.dsf", DSF_HEADER, 1024 * 1024);  // 1MB file
    std::string filePath = testFiles_.back();

    // Act
    auto result = formatDetector_->detectFormat(filePath);

    // Assert
    ASSERT_TRUE(result.has_value()) << "DSD format should be detected";
    EXPECT_EQ(result->format, AudioFormat::DSF) << "Detected format should be DSF";

    // For DSD1024, sample rate should be very high (45.1584 MHz)
    if (result->sampleRate > 1000000) {  // If sample rate is detected correctly
        EXPECT_GE(result->sampleRate, 44100 * 1024) << "Should support high DSD sample rates";
    }
}

// Test format detection with corrupted headers
TEST_F(FormatDetectorTest, DetectCorruptedHeaders) {
    // Test with slightly corrupted WAV header
    std::vector<uint8_t> corruptedWav = WAV_HEADER;
    corruptedWav[4] = 0x00;  // Corrupt one byte in header

    addTestFile("corrupted.wav", corruptedWav);
    std::string filePath = testFiles_.back();

    // Act
    auto result = formatDetector_->detectFormat(filePath);

    // Assert - Should still try to detect from extension or fallback
    ASSERT_TRUE(result.has_value()) << "Should attempt detection even with corrupted header";
}

// Test format detection thread safety
TEST_F(FormatDetectorTest, FormatDetectionThreadSafety) {
    const int numThreads = 10;
    const int filesPerThread = 10;
    std::vector<std::thread> threads;
    std::atomic<int> successfulDetections{0};

    // Arrange - Create test files
    std::vector<std::vector<std::string>> threadFiles(numThreads);
    for (int t = 0; t < numThreads; ++t) {
        for (int f = 0; f < filesPerThread; ++f) {
            std::string filename = "thread_" + std::to_string(t) + "_file_" + std::to_string(f) + ".flac";
            addTestFile(filename, FLAC_HEADER);
            threadFiles[t].push_back(testFiles_.back());
        }
    }

    // Act - Run detection in parallel
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (const auto& filePath : threadFiles[t]) {
                auto result = formatDetector_->detectFormat(filePath);
                if (result.has_value() && result->format == AudioFormat::FLAC) {
                    successfulDetections++;
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Assert - All detections should be successful
    EXPECT_EQ(successfulDetections, numThreads * filesPerThread)
        << "All parallel format detections should succeed";
}

// Test format detection accuracy score
TEST_F(FormatDetectorTest, FormatDetectionAccuracy) {
    // Test various formats and check detection accuracy
    std::map<AudioFormat, std::vector<uint8_t>> formatMap = {
        {AudioFormat::WAV, WAV_HEADER},
        {AudioFormat::FLAC, FLAC_HEADER},
        {AudioFormat::OGG, OGG_HEADER},
        {AudioFormat::M4A, M4A_HEADER},
        {AudioFormat::DFF, DFF_HEADER},
        {AudioFormat::DSF, DSF_HEADER}
    };

    int successfulDetections = 0;
    int totalFormats = formatMap.size();

    for (const auto& [format, header] : formatMap) {
        std::string filename = "accuracy_test." +
            std::filesystem::path("dummy").replace_extension(formatDetector_->getExtensionForFormat(format)).string();
        addTestFile(filename, header);
        std::string filePath = testFiles_.back();

        auto result = formatDetector_->detectFormat(filePath);
        if (result.has_value() && result->format == format) {
            successfulDetections++;
        }
    }

    // Assert - Should detect all supported formats correctly
    EXPECT_EQ(successfulDetections, totalFormats)
        << "Should correctly detect all supported audio formats";
}