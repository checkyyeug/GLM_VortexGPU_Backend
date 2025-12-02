# Multi-stage Dockerfile for Vortex GPU Audio Backend
FROM nvidia/cuda:12.0-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_ARCHITECTURES=86

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    curl \
    wget \
    ca-certificates \
    gnupg \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install audio libraries
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    libpulse-dev \
    libjack-jackd2-dev \
    portaudio19-dev \
    libopus-dev \
    libflac-dev \
    libvorbis-dev \
    libmp3lame-dev \
    libsndfile1-dev \
    libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    intel-mkl \
    mkl-dummy \
    libopenblas-dev \
    liblapack-dev \
    libopencl-dev \
    ocl-icd-opencl-dev \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# Install JUCE dependencies
RUN apt-get update && apt-get install -y \
    freeglut3-dev \
    libx11-dev \
    libxext-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libfreetype6-dev \
    libxft-dev \
    libasound2-dev \
    libjack-jackd2-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . /vortex-backend
WORKDIR /vortex-backend

# Build the project
RUN mkdir -p build && cd build \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
    && make -j$(nproc)

# Production stage
FROM nvidia/cuda:12.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libasound2 \
    libpulseaudio \
    libjack-jackd2 \
    libopus0 \
    libflac8 \
    libvorbisenc2 \
    libmp3lame0 \
    libsndfile1 \
    libfftw3f3 \
    libopencl-1 \
    ocl-icd-opencl-dev \
    vulkan-tools \
    curl \
    ca-certificates \
    && rm -rf /vortex-backend

# Copy built binaries
COPY --from=builder /vortex-backend/build/vortex-backend /usr/local/bin/
COPY --from=builder /vortex-backend/build/lib* /usr/local/lib/
COPY --from=builder /vortex-backend/include /usr/local/include/
COPY --from=builder /vortex-backend/config /etc/vortex/

# Create non-root user
RUN useradd -m -u 1000 vortex && \
    mkdir -p /home/vortex/.vortex && \
    chown -R vortex:vortex /home/vortex

USER vortex
WORKDIR /home/vortex

# Expose ports
EXPOSE 8080 8081 8082

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/system/health || exit 1

# Run the application
CMD ["vortex-backend", "--config", "/etc/vortex/production.json"]