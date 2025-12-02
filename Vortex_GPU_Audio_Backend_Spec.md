# Vortex GPU Audio – 后端架构设计（C++/Rust + GPU加速终极版）

目标：构建一个**全球最强发烧级音频处理引擎**的后端，支持 DSD1024 + 512段EQ + closed-form-16M + 实时GPU加速处理。

## 核心技术栈（2025 最强组合）

### 主要框架
- **C++20/23** (核心音频处理引擎)
- **Rust** (网络服务 + 安全层)
- **CUDA 12.x** + **cuBLAS/cuFFT** (NVIDIA GPU加速)
- **OpenCL** + **Vulkan** (AMD/Intel GPU通用支持)
- **oneAPI** (Intel NPU集成)

### 音频处理库
- **JUCE 8** (专业音频框架)
- **libsndfile** + **FLAC** + **libmp3lame** (格式解码)
- **DSD解码库** (DSD1024支持)
- **Resampler** (高质量采样率转换)
- **Eigen3** + **Intel IPP** (向量化数学)

### 网络通信
- **WebSocket++** (实时数据流)
- **Boost.Beast** (HTTP服务器)
- **ZeroMQ** (进程间通信)
- **Protocol Buffers** (二进制序列化)

### 系统架构
- **CMake** (构建系统)
- **Docker** + **Kubernetes** (容器化部署)
- **Prometheus** + **Grafana** (监控)

## 项目目录结构

```
vortex-backend/
├── src/
│   ├── core/                           # 核心音频处理引擎
│   │   ├── audio_engine.cpp/hpp         # 主音频引擎
│   │   ├── dsp/
│   │   │   ├── eq_processor.cpp/hpp     # 512段EQ处理器
│   │   │   ├── dsd_processor.cpp/hpp    # DSD1024处理器
│   │   │   ├── convolver.cpp/hpp        # closed-form-16M卷积
│   │   │   ├── resampler.cpp/hpp        # 高质量重采样
│   │   │   └── filters/
│   │   │       ├── biquad.cpp/hpp       # 双二阶滤波器
│   │   │       ├── fir_filter.cpp/hpp   # FIR滤波器
│   │   │       └── filter_chain.cpp/hpp # 滤波器链
│   │   ├── gpu/
│   │   │   ├── cuda_processor.cpp/hpp   # CUDA GPU加速
│   │   │   ├── opencl_processor.cpp/hpp # OpenCL通用GPU
│   │   │   ├── vulkan_processor.cpp/hpp # Vulkan计算
│   │   │   └── memory_manager.cpp/hpp   # GPU内存管理
│   │   └── fileio/
│   │       ├── audio_file_loader.cpp/hpp # 音频文件加载器
│   │       ├── format_detector.cpp/hpp   # 格式自动检测
│   │       └── metadata_extractor.cpp/hpp # 元数据提取
│   │
│   ├── network/                        # 网络服务层
│   │   ├── websocket_server.cpp/hpp     # WebSocket实时数据
│   │   ├── http_server.cpp/hpp          # REST API服务器
│   │   ├── discovery_service.cpp/hpp    # 设备自动发现
│   │   ├── protocol/                    # 通信协议
│   │   │   ├── binary_protocol.cpp/hpp  # 二进制协议处理
│   │   │   └── json_protocol.cpp/hpp    # JSON协议处理
│   │   └── authentication.cpp/hpp       # 认证与安全
│   │
│   ├── output/                         # 输出管理
│   │   ├── output_manager.cpp/hpp       # 输出设备管理器
│   │   ├── roon_bridge.cpp/hpp          # Roon桥接
│   │   ├── hqplayer_naa.cpp/hpp         # HQPlayer NAA
│   │   ├── upnp_renderer.cpp/hpp        # UPnP渲染器
│   │   └── local_output.cpp/hpp         # 本地音频输出
│   │
│   ├── system/                         # 系统监控
│   │   ├── hardware_monitor.cpp/hpp     # GPU/NPU/CPU监控
│   │   ├── latency_analyzer.cpp/hpp     # 延迟分析器
│   │   ├── performance_counter.cpp/hpp  # 性能计数器
│   │   └── system_info.cpp/hpp          # 系统信息
│   │
│   └── utils/                          # 工具类
│       ├── logger.cpp/hpp               # 日志系统
│       ├── config_manager.cpp/hpp       # 配置管理
│       ├── thread_pool.cpp/hpp          # 线程池
│       └── math_utils.cpp/hpp           # 数学工具
│
├── include/                            # 公共头文件
│   ├── vortex_api.hpp                   # 核心API定义
│   ├── audio_types.hpp                  # 音频类型定义
│   └── network_types.hpp                # 网络类型定义
│
├── shaders/                            # GPU着色器
│   ├── audio_processing.comp            # 音频处理计算着色器
│   ├── spectrum_analyzer.comp           # 频谱分析着色器
│   └── convolution.comp                 # 卷积运算着色器
│
├── config/                             # 配置文件
│   ├── default.json                     # 默认配置
│   ├── production.json                  # 生产环境配置
│   └── development.json                 # 开发环境配置
│
├── tests/                              # 测试代码
│   ├── unit/                           # 单元测试
│   ├── integration/                    # 集成测试
│   └── performance/                    # 性能测试
│
├── tools/                             # 开发工具
│   ├── benchmark/                      # 性能基准测试
│   ├── profiler/                       # 性能分析器
│   └── diagnostics/                    # 诊断工具
│
├── docs/                              # 文档
│   ├── api/                           # API文档
│   ├── architecture/                  # 架构文档
│   └── deployment/                    # 部署文档
│
├── CMakeLists.txt                     # CMake构建配置
├── Dockerfile                         # Docker配置
├── docker-compose.yml                 # Docker编排
├── package.json                       # 依赖管理
└── README.md                          # 项目说明
```

## 核心API设计

### 1. REST API端点

#### 音频文件处理
```cpp
// 文件上传和格式检测
POST /api/audio/upload
- Content-Type: multipart/form-data
- 支持：音频文件 + 元数据
- 返回：FileUploadProgress + AudioFile信息

// 获取支持格式列表
GET /api/audio/formats
- 返回：SupportedFormat[]

// 获取音频文件元数据
GET /api/audio/{fileId}/metadata
- 返回：AudioMetadata

// 音频文件处理状态
GET /api/audio/{fileId}/status
- 返回：AudioFile.status + 进度信息
```

#### 音频处理控制
```cpp
// 开始/停止播放
POST /api/player/play
POST /api/player/stop
POST /api/player/pause
POST /api/player/seek

// 音量控制
POST /api/player/volume
- body: { volume: 0.0-1.0 }

// 播放位置
GET /api/player/position
- 返回：{ currentTime: number, duration: number }
```

#### 滤波器和处理链
```cpp
// 获取可用滤波器列表
GET /api/filters
- 返回：FilterDefinition[]

// 添加滤波器到处理链
POST /api/chain/add
- body: { filterId: string, position?: number }

// 移除滤波器
DELETE /api/chain/{filterId}

// 调整滤波器参数
PUT /api/chain/{filterId}/parameters
- body: { parameters: Record<string, any> }

// 滤波器控制（bypass/solo/wet）
POST /api/chain/{filterId}/control
- body: { bypass?: boolean, solo?: boolean, wet?: number }
```

#### 输出设备管理
```cpp
// 发现输出设备
GET /api/output/discover
- 返回：OutputDevice[]

// 选择输出设备
POST /api/output/select
- body: { deviceId: string }

// 获取输出设备状态
GET /api/output/status
- 返回：OutputDeviceStatus
```

#### 系统监控
```cpp
// 硬件状态
GET /api/system/hardware
- 返回：HardwareStatus

// 延迟分析
GET /api/system/latency
- 返回：LatencyAnalysis

// 系统信息
GET /api/system/info
- 返回：SystemInfo
```

### 2. WebSocket实时数据协议

#### 连接建立
```javascript
// 客户端连接
ws://localhost:8080/ws

// 认证消息（可选）
{
  type: "auth",
  token: "auth_token"
}
```

#### 实时数据流
```javascript
// 频谱数据（2048点）
{
  type: "spectrum",
  timestamp: 1234567890,
  data: {
    bins: Float32Array(2048),
    frequencyRange: [20, 20000]
  }
}

// 波形数据（4096样本）
{
  type: "waveform",
  timestamp: 1234567890,
  data: {
    left: Float32Array(4096),
    right: Float32Array(4096)
  }
}

// VU/PPM/Peak表
{
  type: "meters",
  timestamp: 1234567890,
  data: {
    vuLeft: number,    // -60 ~ 0 dB
    vuRight: number,
    peakLeft: number,  // 峰值保持
    peakRight: number,
    rmsLeft: number,   // RMS值
    rmsRight: number,
    stereoCorrelation: number  // 立体声相关性
  }
}

// 硬件状态
{
  type: "hardware",
  timestamp: 1234567890,
  data: {
    gpu: {
      usage: number,        // 0-100%
      memoryUsed: number,   // MB
      temperature: number,  // °C
      powerUsage: number    // W
    },
    npu: {
      usage: number,
      memoryUsed: number
    },
    cpu: {
      usage: number,
      cores: number,
      temperature: number
    },
    latency: {
      total: number,        // 总延迟ms
      breakdown: {
        input: number,      // 输入延迟
        processing: number, // 处理延迟
        output: number      // 输出延迟
      }
    }
  }
}

// 滤波器状态
{
  type: "filter_status",
  timestamp: 1234567890,
  data: {
    filterId: string,
    name: string,
    isActive: boolean,
    bypass: boolean,
    solo: boolean,
    wet: number,
    parameters: Record<string, number>
  }
}
```

#### 客户端控制命令
```javascript
// 实时调整参数
{
  type: "set_parameter",
  filterId: string,
  parameter: string,
  value: number
}

// 切换bypass
{
  type: "toggle_bypass",
  filterId: string
}

// 设置wet值
{
  type: "set_wet",
  filterId: string,
  wet: number
}
```

## 音频处理核心架构

### 1. 音频引擎设计

```cpp
class AudioEngine {
public:
    // 初始化音频引擎
    bool Initialize(int sampleRate, int bufferSize);

    // 处理音频块
    void ProcessAudioBlock(float* input, float* output, int numSamples);

    // 添加/移除滤波器
    void AddFilter(std::unique_ptr<Filter> filter);
    void RemoveFilter(const std::string& filterId);

    // GPU加速处理
    void EnableGPUAcceleration(GPUBackend backend);

private:
    std::vector<std::unique_ptr<Filter>> m_filterChain;
    std::unique_ptr<GPUProcessor> m_gpuProcessor;
    ThreadPool m_processingThreads;
};
```

### 2. 滤波器基类设计

```cpp
class Filter {
public:
    virtual ~Filter() = default;

    // 音频处理
    virtual void Process(float* input, float* output, int numSamples) = 0;

    // 参数控制
    virtual void SetParameter(const std::string& name, float value) = 0;
    virtual float GetParameter(const std::string& name) const = 0;

    // 状态控制
    virtual void SetBypass(bool bypass) { m_bypass = bypass; }
    virtual void SetSolo(bool solo) { m_solo = solo; }
    virtual void SetWet(float wet) { m_wet = std::clamp(wet, 0.0f, 1.0f); }

protected:
    bool m_bypass = false;
    bool m_solo = false;
    float m_wet = 1.0f;
    std::string m_filterId;
    std::string m_name;
};
```

### 3. GPU加速处理器

```cpp
class GPUProcessor {
public:
    // 初始化GPU后端
    bool Initialize(GPUBackend backend);

    // 上传音频数据到GPU
    void UploadAudioData(const float* input, int numSamples);

    // GPU音频处理
    void ProcessAudioGPU();

    // 下载处理结果
    void DownloadResults(float* output, int numSamples);

    // 获取GPU信息
    GPUInfo GetGPUInfo() const;

private:
    std::unique_ptr<CUDAProcessor> m_cudaProcessor;
    std::unique_ptr<OpenCLProcessor> m_openclProcessor;
    GPUBuffer m_inputBuffer;
    GPUBuffer m_outputBuffer;
};
```

### 4. DSD1024处理器

```cpp
class DSDProcessor : public Filter {
public:
    DSDProcessor();

    // DSD解码处理
    void Process(float* input, float* output, int numSamples) override;

    // 设置DSD参数
    void SetDSDMode(DSDMode mode);  // DSD64, DSD128, DSD256, DSD512, DSD1024
    void SetModulationFrequency(float freq);  // 调制频率

private:
    DSDMode m_mode = DSDMode::DSD1024;
    float m_modFreq = 2.8224e6f;  // DSD1024标准频率
    std::unique_ptr<PDMDecoder> m_pdmDecoder;
};
```

### 5. 512段EQ处理器

```cpp
class EqualizerProcessor : public Filter {
public:
    EqualizerProcessor();

    // EQ处理
    void Process(float* input, float* output, int numSamples) override;

    // 设置EQ频段
    void SetBand(int bandIndex, float frequency, float gain, float q);

    // 设置EQ曲线类型
    void SetCurveType(EQCurveType type);  // Graphic, Parametric, Shelving

private:
    static constexpr int NUM_BANDS = 512;
    std::array<BiquadFilter, NUM_BANDS> m_bands;
    FFTProcessor m_fftProcessor;
};
```

### 6. 16M点卷积处理器

```cpp
class ConvolutionProcessor : public Filter {
public:
    ConvolutionProcessor();

    // 卷积处理
    void Process(float* input, float* output, int numSamples) override;

    // 加载脉冲响应
    bool LoadImpulseResponse(const std::string& irPath);

    // 设置卷积模式
    void SetConvolutionMode(ConvolutionMode mode);  // Direct, Partitioned, Multithread

private:
    static constexpr int IR_LENGTH = 16777216;  // 16M samples
    std::unique_ptr<FFTConvolver> m_convolver;
    GPUBuffer m_irBuffer;  // GPU上的IR数据
};
```

## 文件格式支持

### 支持的音频格式
```cpp
enum class AudioFormat {
    // 无损格式
    WAV_PCM,        // WAV (PCM)
    WAV_FLOAT,      // WAV (32-bit float)
    FLAC,           // FLAC
    ALAC,           // Apple Lossless

    // 有损格式
    MP3,            // MPEG-1/2/2.5 Layer 3
    AAC,            // AAC/MP4
    OGG_VORBIS,     // OGG Vorbis
    M4A,            // Apple M4A

    // 高分辨率格式
    DSD64,          // DSD 2.8224 MHz
    DSD128,         // DSD 5.6448 MHz
    DSD256,         // DSD 11.2896 MHz
    DSD512,         // DSD 22.5792 MHz
    DSD1024,        // DSD 45.1584 MHz
    DSF,            // DSD Stream File
    DFF             // DSDIFF
};
```

### 文件加载器接口
```cpp
class AudioFileLoader {
public:
    // 检测文件格式
    static AudioFormat DetectFormat(const std::string& filePath);

    // 加载音频文件
    std::unique_ptr<AudioData> LoadFile(const std::string& filePath,
                                       LoadingProgressCallback callback = nullptr);

    // 获取文件元数据
    AudioMetadata GetMetadata(const std::string& filePath);

    // 验证文件完整性
    bool ValidateFile(const std::string& filePath);

private:
    std::map<AudioFormat, std::unique_ptr<FormatDecoder>> m_decoders;
};
```

## 输出设备支持

### 输出设备类型
```cpp
enum class OutputDeviceType {
    Local,          // 本地音频设备
    RoonBridge,     // Roon Bridge
    HQPlayerNAA,    // HQPlayer NAA
    UPnPRenderer,   // UPnP/DLNA渲染器
    NetworkStream   // 网络流
};
```

### 输出管理器
```cpp
class OutputManager {
public:
    // 发现输出设备
    std::vector<OutputDevice> DiscoverDevices();

    // 连接到设备
    bool ConnectToDevice(const std::string& deviceId);

    // 发送音频数据
    bool SendAudioData(const float* audioData, int numSamples);

    // 设备状态监控
    OutputDeviceStatus GetDeviceStatus(const std::string& deviceId);

private:
    std::map<std::string, std::unique_ptr<OutputDevice>> m_devices;
    std::unique_ptr<RoonBridgeClient> m_roonClient;
    std::unique_ptr<HQPlayerNAA> m_hqpClient;
    std::unique_ptr<UPnPRenderer> m_upnpRenderer;
};
```

## 性能优化策略

### 1. GPU内存管理
- 使用内存池避免频繁分配
- 异步内存传输提高吞吐量
- 统一内存架构减少数据拷贝

### 2. 多线程处理
- 音频处理线程池
- 网络I/O异步处理
- GPU/CPU并行计算

### 3. 缓存策略
- 频谱分析结果缓存
- 滤波器系数缓存
- 文件元数据缓存

### 4. 延迟优化
- 零拷贝音频缓冲区
- 实时优先级线程
- GPU直接内存访问

## 配置管理

### 主配置文件 (config/default.json)
```json
{
  "audio": {
    "sampleRate": 768000,
    "bitDepth": 32,
    "bufferSize": 4096,
    "channels": 2,
    "enableGPU": true,
    "gpuBackend": "cuda"
  },
  "processing": {
    "enableDSD1024": true,
    "eqBands": 512,
    "convolutionLength": 16777216,
    "enableMultithreading": true,
    "threadPoolSize": 8
  },
  "network": {
    "port": 8080,
    "websocketPort": 8081,
    "enableSSL": false,
    "maxConnections": 100,
    "discoveryPort": 8082
  },
  "output": {
    "defaultDevice": "local",
    "enableAutoDiscovery": true,
    "supportedFormats": ["DSD1024", "PCM768k"]
  },
  "monitoring": {
    "enablePrometheus": true,
    "enableLatencyAnalysis": true,
    "reportingInterval": 1000
  }
}
```

## 部署架构

### Docker容器化
```dockerfile
FROM nvidia/cuda:12.0-devel-ubuntu22.04

# 安装依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libjuce-dev \
    libasound2-dev \
    libpulse-dev

# 编译安装
COPY . /vortex-backend
WORKDIR /vortex-backend
RUN cmake . && make -j$(nproc)

# 运行配置
EXPOSE 8080 8081 8082
CMD ["./vortex-backend", "--config", "config/production.json"]
```

### Kubernetes部署
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vortex-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vortex-backend
  template:
    metadata:
      labels:
        app: vortex-backend
    spec:
      containers:
      - name: vortex-backend
        image: vortex-backend:latest
        ports:
        - containerPort: 8080
        - containerPort: 8081
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
```

## 监控和诊断

### Prometheus指标
- 音频处理延迟
- GPU利用率
- 内存使用量
- 网络吞吐量
- 错误率统计

### 健康检查端点
```cpp
GET /health
- 返回系统整体状态

GET /health/audio
- 返回音频处理状态

GET /health/gpu
- 返回GPU状态
```

这个后端设计文档与前端的Vue3规范完美匹配，支持：

✅ **DSD1024 + 512段EQ + 16M卷积** 超高分辨率处理
✅ **GPU/NPU加速** 实时音频处理
✅ **全格式音频支持** 从MP3到DSD1024
✅ **WebSocket实时数据** 频谱/波形/VU表
✅ **多输出设备** Roon/HQPlayer/UPnP集成
✅ **高性能架构** 容器化部署 + 监控

这是全球发烧友梦寐以求的终极音频处理系统！