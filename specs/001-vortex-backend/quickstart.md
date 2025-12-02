# Quickstart Guide: Vortex GPU Audio Backend

**Purpose**: Rapid development setup and first-time user guide
**Created**: 2025-12-01
**Feature**: [Vortex GPU Audio Backend](spec.md)

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA 12.x support (recommended) or AMD/Intel GPU with OpenCL support
- **CPU**: Multi-core processor (8+ cores recommended for real-time processing)
- **Memory**: 16GB RAM minimum, 32GB recommended for large audio files
- **Storage**: SSD with at least 10GB free space for audio files and processing
- **Audio**: Professional audio interface or high-quality onboard audio

### Software Requirements
- **Operating System**: Ubuntu 20.04+ / Windows 10+ / macOS 11+
- **GPU Drivers**: NVIDIA CUDA Toolkit 12.x or latest AMD/Intel GPU drivers
- **Build Tools**: CMake 3.20+, GCC 10+/Clang 12+, or Visual Studio 2022
- **Runtime**: Docker 20.10+ (optional, for containerized deployment)

## Quick Installation

### Option 1: Docker (Recommended for Testing)
```bash
# Clone the repository
git clone https://github.com/vortexaudio/vortex-backend.git
cd vortex-backend

# Build and run with Docker
docker build -t vortex-backend .
docker run -p 8080:8080 -p 8081:8081 --gpus all vortex-backend

# Verify installation
curl http://localhost:8080/api/system/health
```

### Option 2: Native Build (Development)
```bash
# Install dependencies (Ubuntu)
sudo apt update
sudo apt install -y cmake build-essential \
    libjuce-dev libasound2-dev libpulse-dev \
    nvidia-cuda-dev nvidia-opencl-dev

# Build from source
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run the application
./vortex-backend --config ../config/production.json
```

### Option 3: Package Installation (Production)
```bash
# Download and install .deb package (Ubuntu/Debian)
wget https://releases.vortexaudio.com/vortex-backend_1.0.0_amd64.deb
sudo dpkg -i vortex-backend_1.0.0_amd64.deb

# Or .rpm package (CentOS/RHEL)
wget https://releases.vortexaudio.com/vortex-backend-1.0.0.x86_64.rpm
sudo rpm -i vortex-backend-1.0.0.x86_64.rpm

# Start the service
sudo systemctl start vortex-backend
sudo systemctl enable vortex-backend
```

## Basic Configuration

### Environment Setup
```bash
# Copy default configuration
cp config/default.json config/local.json

# Edit configuration for your system
nano config/local.json
```

### Key Configuration Settings
```json
{
  "audio": {
    "sampleRate": 768000,
    "bitDepth": 32,
    "bufferSize": 4096,
    "enableGPU": true,
    "gpuBackend": "cuda"
  },
  "network": {
    "httpPort": 8080,
    "websocketPort": 8081,
    "maxConnections": 100
  },
  "output": {
    "defaultDevice": "local",
    "enableAutoDiscovery": true
  }
}
```

## First Steps

### 1. Verify System Health
```bash
# Check system status
curl http://localhost:8080/api/system/health

# Expected response
{
  "status": "healthy",
  "uptime": 12345,
  "version": "1.0.0",
  "components": {
    "audio_engine": "healthy",
    "gpu_processor": "healthy",
    "network_server": "healthy"
  }
}
```

### 2. Check GPU Capabilities
```bash
# Get hardware status
curl http://localhost:8080/api/system/hardware

# Expected response (GPU detected)
{
  "gpu": {
    "utilization": 15.2,
    "memoryUsed": 512,
    "temperature": 45,
    "gpu_name": "NVIDIA RTX 4090"
  }
}
```

### 3. Upload First Audio File
```bash
# Upload audio file
curl -X POST -F "file=@test_audio.flac" \
  http://localhost:8080/api/audio/upload

# Expected response
{
  "fileId": "uuid-1234-5678-90ab",
  "name": "test_audio.flac",
  "status": "processing",
  "format": {
    "extension": "flac",
    "name": "FLAC"
  }
}
```

### 4. Monitor Processing Progress
```bash
# Check file status
curl http://localhost:8080/api/audio/uuid-1234-5678-90ab/status

# Expected response
{
  "fileId": "uuid-1234-5678-90ab",
  "status": "ready",
  "progress": 1.0
}
```

### 5. Start Playback
```bash
# Begin playback
curl -X POST http://localhost:8080/api/player/play \
  -H "Content-Type: application/json" \
  -d '{"fileId": "uuid-1234-5678-90ab"}'

# Expected response
{
  "state": "playing",
  "fileId": "uuid-1234-5678-90ab",
  "position": 0.0,
  "volume": 0.8
}
```

## WebSocket Integration

### JavaScript Client Example
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8081/ws');

// Subscribe to real-time data
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    dataTypes: ['spectrum', 'waveform', 'meters', 'hardware'],
    updateRate: 60
  }));
};

// Handle real-time spectrum data
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  if (message.type === 'spectrum') {
    // Process 2048-point spectrum data
    const spectrum = message.data;
    console.log('Frequency bins:', spectrum.bins.length);
  }

  if (message.type === 'hardware') {
    // Update GPU utilization display
    const gpu = message.data.gpu;
    console.log('GPU Usage:', gpu.utilization + '%');
  }
};
```

### Python Client Example
```python
import asyncio
import websockets
import json

async def vortex_client():
    uri = "ws://localhost:8081/ws"
    async with websockets.connect(uri) as websocket:
        # Subscribe to real-time data
        subscribe_msg = {
            "type": "subscribe",
            "dataTypes": ["spectrum", "waveform"],
            "updateRate": 30
        }
        await websocket.send(json.dumps(subscribe_msg))

        # Process real-time data
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data['type']} data")

asyncio.run(vortex_client())
```

## Audio Processing Chain

### Create Basic EQ Chain
```bash
# Add 512-band EQ to chain
curl -X POST http://localhost:8080/api/chain/add \
  -H "Content-Type: application/json" \
  -d '{"filterId": "equalizer_512_band", "position": 0}'

# Set EQ parameters
curl -X PUT http://localhost:8080/api/chain/equalizer_512_band/parameters \
  -H "Content-Type: application/json" \
  -d '{"parameters": {"band_100": 3.0, "band_1000": -2.5, "band_10000": 1.0}}'
```

### Add Convolution Reverb
```bash
# Add convolution filter
curl -X POST http://localhost:8080/api/chain/add \
  -H "Content-Type: application/json" \
  -d '{"filterId": "convolution_reverb", "position": 1}'

# Load impulse response
curl -X PUT http://localhost:8080/api/chain/convolution_reverb/parameters \
  -H "Content-Type: application/json" \
  -d '{"parameters": {"ir_file": "hall_impulse.wav", "wet_mix": 0.3}}'
```

## Multi-Device Audio Output

### Discover Output Devices
```bash
# Scan for network devices
curl http://localhost:8080/api/output/discover

# Expected response
[
  {
    "id": "local_default",
    "name": "Default Audio Device",
    "type": "local",
    "supportedFormats": ["PCM768k", "DSD1024"]
  },
  {
    "id": "roon_bridge_1",
    "name": "Living Room Roon Bridge",
    "type": "roon",
    "ipAddress": "192.168.1.100"
  }
]
```

### Switch Output Device
```bash
# Select Roon Bridge output
curl -X POST http://localhost:8080/api/output/select \
  -H "Content-Type: application/json" \
  -d '{"deviceId": "roon_bridge_1"}'
```

## Performance Optimization

### GPU Optimization
```json
{
  "audio": {
    "gpuBackend": "cuda",
    "enableGPU": true
  },
  "gpu": {
    "memoryLimit": 8192,
    "utilizationTarget": 85,
    "enableMultiGPU": true
  }
}
```

### Latency Optimization
```json
{
  "audio": {
    "bufferSize": 256,
    "maxLatency": 5.0
  },
  "network": {
    "websocketBuffer": 16
  }
}
```

## Testing

### Performance Benchmarks
```bash
# Run performance tests
./vortex-backend --benchmark

# Expected output
GPU FFT (2048-point): 1.2ms
512-band EQ: 2.8ms
16M Convolution: 4.1ms
Total Latency: 8.1ms
GPU Utilization: 82%
```

### Stress Testing
```bash
# Continuous playback test
curl -X POST http://localhost:8080/api/test/stress \
  -H "Content-Type: application/json" \
  -d '{"duration": 3600, "concurrent_streams": 4}'
```

## Troubleshooting

### Common Issues

**GPU Not Detected**
```bash
# Check CUDA installation
nvidia-smi

# Verify GPU drivers
nvcc --version
```

**Audio Dropouts**
```bash
# Check system latency
cat /proc/asound/card0/pcm0p/sub0/status

# Increase buffer size
# Edit config/local.json:
# "bufferSize": 8192
```

**High Memory Usage**
```bash
# Monitor memory usage
./vortex-backend --monitor

# Reduce memory limits
# Edit config/local.json:
# "memoryLimit": 4096
```

### Log Files
```bash
# Application logs
tail -f logs/vortex-backend.log

# GPU processing logs
tail -f logs/gpu-processing.log

# Network connection logs
tail -f logs/websocket.log
```

## Next Steps

1. **API Documentation**: Visit `/api/docs` for interactive API documentation
2. **Configuration Guide**: See `docs/configuration.md` for advanced settings
3. **Performance Tuning**: Review `docs/performance.md` for optimization tips
4. **Integration Examples**: Check `examples/` directory for integration samples

## Support

- **Documentation**: https://docs.vortexaudio.com
- **Community**: https://community.vortexaudio.com
- **Issues**: https://github.com/vortexaudio/vortex-backend/issues
- **Email**: support@vortexaudio.com