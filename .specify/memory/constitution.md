<!--
Sync Impact Report:
- Version change: N/A → 1.0.0
- Modified principles: All placeholder principles replaced with TDD-focused principles
- Added sections: Audio Processing Standards, Development Workflow
- Removed sections: None (template placeholders properly replaced)
- Templates requiring updates:
  ✅ plan-template.md (already has TDD references)
  ✅ tasks-template.md (already enforces test-first approach)
  ✅ spec-template.md (already requires independent testing)
- Follow-up TODOs: None
-->

# Vortex GPU Audio Backend Constitution

## Core Principles

### I. Test-First Development (NON-NEGOTIABLE)
All code MUST be written using Test-Driven Development methodology. Tests are written BEFORE implementation code, must FAIL initially, and only then implementation is provided to make them pass. Red-Green-Refactor cycle is mandatory for every feature, bug fix, and optimization. No exceptions allowed for urgent fixes, prototypes, or experimental code.

### II. Audio Processing Contracts
Every audio processing module MUST define and maintain explicit contracts specifying input/output formats, sample rate requirements, bit depth expectations, and processing latency guarantees. GPU/NPU accelerated components must include fallback implementations with equivalent behavior. All contracts must be independently testable with synthetic audio data.

### III. Performance with Verification
Performance optimizations MUST be validated through benchmarks with measurable targets before implementation. Real-time audio processing requires <10ms processing latency, GPU utilization >80% when available, and memory usage predictable within bounds. Every optimization must include regression tests to prevent performance degradation.

### IV. Format Extensibility
Audio format support MUST follow an extensible architecture where new formats (DSD1024, PCM768k, etc.) can be added without modifying existing processing pipelines. Format detection, metadata extraction, and conversion modules must be independently testable with format-specific test suites.

### V. Real-time Reliability
System MUST maintain real-time audio processing guarantees under all load conditions. Audio buffer underruns are unacceptable. WebSocket streaming, GPU processing, and file I/O must never block real-time audio threads. All components must handle failures gracefully and maintain audio continuity.

## Audio Processing Standards

### GPU Acceleration Requirements
All computationally intensive audio processing (FFT, convolution, EQ filtering) MUST utilize GPU acceleration when available. Implementations must detect GPU capabilities (CUDA/OpenCL/Vulkan) and automatically select optimal backend. GPU memory management must prevent allocation failures during real-time processing.

### Cross-Platform Compatibility
Backend MUST support Linux server deployment, Windows development environments, and containerized execution. Audio processing algorithms must produce identical results across platforms within floating-point precision tolerance. Platform-specific optimizations must be encapsulated behind unified interfaces.

### Network Protocol Guarantees
WebSocket streaming MUST deliver real-time audio data with <50ms end-to-end latency. Binary protocol must be frame-synchronized and support recovery from network interruptions. REST API endpoints must complete within 100ms for non-blocking operations. All network failures must be logged and trigger automatic reconnection.

## Development Workflow

### TDD Enforcement Checklist
Every feature implementation requires:
1. **Unit Tests**: Individual component tests covering all edge cases with mock dependencies
2. **Integration Tests**: Real audio data flow through multiple components
3. **Performance Tests**: Latency, throughput, and resource usage benchmarks
4. **Contract Tests**: API interface compatibility and format conformance
5. **GPU Tests**: Both hardware and fallback path validation

### Code Review Requirements
All changes MUST pass automated test coverage with >95% line coverage. Reviews must verify TDD compliance, performance benchmarks, and real-time audio guarantees. GPU code changes require validation on multiple hardware vendors. Audio processing changes require spectral analysis to verify signal integrity.

### Quality Gates
Deployment to production requires:
- All tests passing in continuous integration
- Performance benchmarks meeting targets
- GPU utilization efficiency >80% under load
- Memory usage stable over 24-hour stress test
- Real-time latency guarantees maintained under peak load
- Audio format compatibility validated with test corpus

## Governance

This constitution supersedes all other development practices and coding standards. Amendments require formal documentation, team approval, and migration plan for existing code. All pull requests and code reviews must verify constitutional compliance. Technical complexity must be justified with performance benchmarks and user impact analysis. Use project README and API documentation for runtime development guidance.

**Version**: 1.0.0 | **Ratified**: 2025-12-01 | **Last Amended**: 2025-12-01
