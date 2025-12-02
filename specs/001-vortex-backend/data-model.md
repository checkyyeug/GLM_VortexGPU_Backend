# Data Model: Vortex GPU Audio Backend

**Purpose**: Core data entities and relationships for the audio processing system
**Created**: 2025-12-01
**Feature**: [Vortex GPU Audio Backend](spec.md)

## Core Entities

### AudioFile

Represents uploaded audio files with comprehensive metadata and processing state.

```cpp
struct AudioFile {
    string id;                           // Unique identifier
    string name;                         // Original filename
    string filePath;                     // File system path
    AudioFormat format;                  // Audio format enumeration
    AudioMetadata metadata;              // Technical and artistic metadata
    ProcessingStatus status;             // Current processing state
    float loadingProgress;               // 0.0-1.0 completion percentage
    uint64_t fileSize;                   // File size in bytes
    Duration duration;                   // Audio duration in seconds
    SystemClock::time_point createdAt;   // Upload timestamp
    SystemClock::time_point updatedAt;   // Last modification timestamp
    optional<string> errorMessage;       // Error details if status == ERROR
};
```

**Validation Rules**:
- `id`: Required, UUID v4 format
- `name`: Required, max 255 characters
- `filePath`: Required, valid file system path
- `format`: Required, must be supported format
- `loadingProgress`: 0.0-1.0 inclusive
- `fileSize`: Positive integer

**State Transitions**:
```
IDLE → LOADING → READY → PROCESSING → COMPLETED
                    ↓
                  ERROR → IDLE (retry)
```

### AudioMetadata

Comprehensive metadata extracted from audio files.

```cpp
struct AudioMetadata {
    // Technical metadata
    AudioFormat format;
    string codec;
    uint32_t bitrate;              // kbps
    uint32_t sampleRate;           // Hz (44100, 48000, 96000, 192000, 384000, 768000)
    uint16_t bitDepth;             // 16, 24, 32
    uint16_t channels;             // 1, 2, 6, 8
    Duration duration;             // seconds

    // Artistic metadata
    optional<string> title;
    optional<string> artist;
    optional<string> album;
    optional<uint16_t> year;
    optional<string> genre;
    optional<uint16_t> track;
    optional<string> albumArt;     // Base64 or URL

    // Quality metrics
    float dynamicRange;            // LUFS
    float peakLevel;               // dBFS
    optional<float> replayGain;    // dB
};
```

### ProcessingChain

Ordered sequence of audio processing filters with real-time control.

```cpp
struct ProcessingChain {
    string id;
    string name;
    vector<FilterModule> filters;
    bool isActive;
    float masterVolume;            // 0.0-1.0
    bool bypassChain;
    SystemClock::time_point lastModified;
};

struct FilterModule {
    string id;
    string name;
    FilterType type;
    bool isEnabled;
    bool isBypassed;
    bool isSolo;
    float wetMix;                  // 0.0-1.0
    map<string, float> parameters; // Dynamic parameter values
    uint32_t position;             // Order in chain
    optional<string> pluginId;     // External plugin reference
};
```

**Filter Types**:
```cpp
enum class FilterType {
    EQUALIZER,          // 512-band parametric EQ
    CONVOLUTION,        // 16M-point impulse response
    DSD_PROCESSOR,      // DSD decoding and processing
    RESAMPLER,         // Sample rate conversion
    COMPRESSOR,        // Dynamic range processing
    LIMITER,           // Peak limiting
    DELAY,             // Time-based effects
    REVERB,            // Spatial processing
    GATE,              // Noise gating
    CUSTOM             // User-defined processing
};
```

### OutputDevice

Network or local audio destination with capability detection.

```cpp
struct OutputDevice {
    string id;
    string name;
    OutputDeviceType type;
    string ipAddress;              // For network devices
    uint16_t port;                // Network port
    vector<AudioFormat> supportedFormats;
    vector<uint32_t> supportedSampleRates;
    vector<uint16_t> supportedBitDepths;
    uint32_t maxChannels;
    uint32_t latency;              // ms
    bool isConnected;
    DeviceCapabilities capabilities;
    SystemClock::time_point lastSeen;
};

struct DeviceCapabilities {
    bool supportsDSD;             // DSD64-1024 support
    bool supportsPCM768;          // 768kHz PCM support
    bool supportsMultiChannel;    // >2 channel support
    bool supportsRealtimeControl; // Parameter adjustment during playback
    uint32_t maxBitrate;          // Maximum supported bitrate
    vector<string> exclusiveFeatures; // Unique device capabilities
};
```

**Device Types**:
```cpp
enum class OutputDeviceType {
    LOCAL,              // System audio device
    ROON_BRIDGE,        // Roon Bridge endpoint
    HQPLAYER_NAA,       // HQPlayer NAA endpoint
    UPNP_RENDERER,      // UPnP/DLNA renderer
    NETWORK_STREAM      // Custom network streaming
};
```

### AudioSession

Active audio processing session with real-time state.

```cpp
struct AudioSession {
    string id;
    optional<string> audioFileId;  // Currently loaded file
    optional<string> outputDeviceId; // Current output destination
    optional<string> processingChainId; // Active processing chain
    PlaybackState state;
    Duration currentTime;          // Current playback position
    float volume;                  // 0.0-1.0
    bool isMuted;
    PlaybackMode mode;
    SystemClock::time_point sessionStart;
    SystemClock::time_point lastActivity;
};

enum class PlaybackState {
    STOPPED,
    PLAYING,
    PAUSED,
    SEEKING,
    BUFFERING,
    ERROR
};

enum class PlaybackMode {
    PLAY_ONCE,
    LOOP_TRACK,
    LOOP_PLAYLIST,
    SHUFFLE
};
```

### RealTimeData

Live audio processing metrics and visualization data.

```cpp
struct RealTimeData {
    uint64_t timestamp;            // UNIX epoch milliseconds

    // Audio visualization data
    SpectrumData spectrum;         // 2048-point frequency analysis
    WaveformData waveform;         // 4096-sample time domain
    VUMeterData vuMeters;          // Level measurement

    // Hardware monitoring
    HardwareStatus hardware;       // GPU/NPU/CPU utilization
    ProcessingMetrics processing;  // Audio processing performance

    // System status
    ConnectionStatus network;      // WebSocket connection health
    vector<SystemAlert> alerts;    // Active system warnings
};

struct SpectrumData {
    vector<float> bins;            // 2048 frequency bins (20Hz-20kHz)
    float frequencyRange[2];       // [min, max] frequency range
    float amplitudeRange[2];       // [min, max] amplitude range
    uint32_t fftSize;              // FFT size (2048)
    float windowOverlap;           // Overlap factor (0.75 typical)
};

struct WaveformData {
    vector<float> leftChannel;     // 4096 samples
    vector<float> rightChannel;    // 4096 samples
    uint32_t sampleRate;           // Current sample rate
    uint16_t bitDepth;             // Current bit depth
    float peakLevels[2];           // [left, right] peak levels
};

struct VUMeterData {
    float vuLevels[2];             // [left, right] VU levels (-60 to 0 dB)
    float peakLevels[2];           // [left, right] peak hold levels
    float rmsLevels[2];            // [left, right] RMS levels
    float stereoCorrelation;       // Mono compatibility measure
    float dynamicRange;            // Current dynamic range
};

struct HardwareStatus {
    GPUStatus gpu;
    NPUStatus npu;
    CPUStatus cpu;
    MemoryStatus memory;
    LatencyAnalysis latency;
};

struct GPUStatus {
    float utilization;             // 0-100%
    uint64_t memoryUsed;           // MB
    uint64_t memoryTotal;          // MB
    float temperature;             // Celsius
    float powerUsage;              // Watts
    float clockSpeed;              // MHz
    uint32_t activeCores;          // Cores in use
};

struct ProcessingMetrics {
    float processingLatency;       // ms
    float throughput;              // samples/second
    uint32_t droppedFrames;        // Count of buffer underruns
    float cpuUsage;                // Audio processing thread %
    uint64_t samplesProcessed;     // Total samples processed
    float qualityScore;            // 0-100 signal quality assessment
};
```

### SystemConfiguration

Global system settings and user preferences.

```cpp
struct SystemConfiguration {
    AudioSettings audio;
    NetworkSettings network;
    GPUPreferences gpu;
    MonitoringSettings monitoring;
    UserPreferences user;
    SystemClock::time_point lastModified;
};

struct AudioSettings {
    uint32_t defaultSampleRate;    // 44100, 48000, 96000, 192000, 384000, 768000
    uint16_t defaultBitDepth;      // 16, 24, 32
    uint16_t defaultChannels;      // 1, 2, 6, 8
    uint32_t bufferSize;           // Audio buffer size in samples
    float maxLatency;              // Maximum acceptable latency (ms)
    bool enableDSDProcessing;      // DSD1024 support toggle
    bool enableGPUAcceleration;    // GPU processing toggle
};

struct NetworkSettings {
    uint16_t httpPort;             // HTTP API port (default: 8080)
    uint16_t websocketPort;        // WebSocket port (default: 8081)
    uint16_t discoveryPort;        // Device discovery port (default: 8082)
    bool enableSSL;                // HTTPS/WSS support
    uint32_t maxConnections;       // Maximum concurrent clients
    uint32_t connectionTimeout;    // Connection timeout (seconds)
};

struct GPUPreferences {
    vector<GPUBackend> preferredBackends; // CUDA, OpenCL, Vulkan priority order
    bool enableMultiGPU;           // Use multiple GPUs if available
    uint64_t memoryLimit;          // GPU memory limit (MB)
    float utilizationTarget;       // Target GPU utilization (0-100%)
    bool enableFallback;           // CPU fallback on GPU failure
};
```

## Relationships

```
AudioSession 1--0..1 AudioFile (loaded file)
AudioSession 1--0..1 OutputDevice (current output)
AudioSession 1--0..1 ProcessingChain (active chain)

ProcessingChain 1--* FilterModule (ordered sequence)
FilterModule *--1 FilterParameters (dynamic values)

OutputDevice 1--* AudioCapability (supported formats/modes)
OutputDevice 1--* DeviceStatus (connection monitoring)

RealTimeData 1--1 HardwareStatus (system monitoring)
RealTimeData 1--1 SpectrumData (frequency analysis)
RealTimeData 1--1 WaveformData (time domain)
RealTimeData 1--1 VUMeterData (level measurement)

SystemConfiguration 1--1 AudioSettings
SystemConfiguration 1--1 NetworkSettings
SystemConfiguration 1--1 GPUPreferences
```

## Data Validation

### Type Safety
- All string fields have length limits and encoding validation
- Numeric ranges enforced (0-1 for normalized values, etc.)
- Enum values validated against defined sets
- Timestamp validation for temporal data

### Consistency Rules
- AudioFile format matches detected codec capabilities
- OutputDevice capabilities match actual device support
- FilterModule parameters are within valid ranges for filter type
- RealTimeData timestamps are monotonically increasing

### Performance Considerations
- RealTimeData optimized for frequent updates (minimal validation overhead)
- Large audio data structures use memory pools and zero-copy patterns
- Immutable metadata cached after initial extraction
- Configuration changes validated before application