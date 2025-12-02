---

description: "Task list for Vortex GPU Audio Backend implementation"
---

# Tasks: Vortex GPU Audio Backend

**Input**: Design documents from `/specs/001-vortex-backend/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: This feature requires comprehensive testing per the TDD constitution:
- Unit tests for all audio processing components
- Integration tests for real-time audio pipeline
- Performance tests for latency and GPU utilization
- Contract tests for API interface compliance
- GPU tests for both hardware and fallback paths

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story

## Format: `[ID] [P?] [Story?] Description with file path`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Single backend project**: `vortex-backend/src/`, `tests/` at repository root
- All paths are relative to the repository root
- GPU shaders in `vortex-backend/shaders/`
- Configuration files in `vortex-backend/config/`
- Build system at repository root (`CMakeLists.txt`, `Dockerfile`)

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create vortex-backend project structure per implementation plan
- [ ] T002 Initialize C++20/23 project with JUCE 8 framework and dependencies
- [ ] T003 [P] Setup Rust network services with Cargo.toml for WebSocket and HTTP components
- [ ] T004 [P] Configure CMake build system for hybrid C++/Rust architecture
- [ ] T005 [P] Create Docker configuration for GPU-accelerated container deployment
- [ ] T006 Setup GoogleTest framework for C++ unit testing
- [ ] T007 [P] Setup cargo test framework for Rust network services
- [ ] T008 [P] Create custom audio processing test harness for real-time testing
- [ ] T009 [P] Configure GPU development environment (CUDA 12.x, OpenCL, Vulkan)
- [ ] T010 [P] Setup CI/CD pipeline with GPU testing capabilities

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Core Audio Engine Foundation
- [ ] T011 [P] Create core audio engine interface in include/vortex_api.hpp
- [ ] T012 [P] Create audio type definitions in include/audio_types.hpp
- [ ] T013 [P] Create network type definitions in include/network_types.hpp
- [ ] T014 [P] Implement base audio engine class in src/core/audio_engine.cpp/hpp
- [ ] T015 [P] Create GPU processor base class in src/core/gpu/gpu_processor.cpp/hpp
- [ ] T016 [P] Implement GPU memory manager in src/core/gpu/memory_manager.cpp/hpp

### GPU Backend Implementation
- [ ] T017 [P] Implement CUDA processor in src/core/gpu/cuda_processor.cpp/hpp
- [ ] T018 [P] Implement OpenCL processor in src/core/gpu/opencl_processor.cpp/hpp
- [ ] T019 [P] Implement Vulkan processor in src/core/gpu/vulkan_processor.cpp/hpp
- [ ] T020 [P] Create GPU audio processing compute shader in shaders/audio_processing.comp

### Audio I/O Foundation
- [ ] T021 [P] Implement audio file loader base class in src/core/fileio/audio_file_loader.cpp/hpp
- [ ] T022 [P] Create format detector in src/core/fileio/format_detector.cpp/hpp
- [ ] T023 [P] Implement metadata extractor in src/core/fileio/metadata_extractor.cpp/hpp

### Network Infrastructure
- [ ] T024 [P] Create HTTP server foundation in src/network/http_server.cpp/hpp
- [ ] T025 [P] Create WebSocket server foundation in src/network/websocket_server.cpp/hpp
- [ ] T026 [P] Implement binary protocol handler in src/network/protocol/binary_protocol.cpp/hpp
- [ ] T027 [P] Create Protocol Buffers integration for metadata serialization

### System Monitoring Foundation
- [ ] T028 [P] Create hardware monitor in src/system/hardware_monitor.cpp/hpp
- [ ] T029 [P] Implement latency analyzer in src/system/latency_analyzer.cpp/hpp
- [ ] T030 [P] Create performance counter in src/system/performance_counter.cpp/hpp

### Utility Infrastructure
- [ ] T031 [P] Create logging system in src/utils/logger.cpp/hpp
- [ ] T032 [P] Implement configuration manager in src/utils/config_manager.cpp/hpp
- [ ] T033 [P] Create thread pool in src/utils/thread_pool.cpp/hpp

### Foundation Testing
- [ ] T034 [P] Create unit test for GPU processor initialization in tests/unit/test_gpu_processor.cpp
- [ ] T035 [P] Create unit test for audio engine initialization in tests/unit/test_audio_engine.cpp
- [ ] T036 [P] Create integration test for WebSocket binary protocol in tests/integration/test_websocket_protocol.cpp
- [ ] T037 [P] Create performance test for GPU memory management in tests/performance/test_gpu_memory.cpp

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Audio File Processing (Priority: P1) 🎯 MVP

**Goal**: Enable upload, format detection, metadata extraction, and processing of high-resolution audio files

**Independent Test**: Can be fully tested by uploading various audio formats and verifying they are processed with correct metadata extraction and format conversion

### Tests for User Story 1 (TDD Required) ⚠️

> **NOTE**: Write these tests FIRST, ensure they FAIL before implementation

- [ ] T038 [P] [US1] Contract test for audio file upload endpoint in tests/contract/test_audio_upload.cpp
- [ ] T039 [P] [US1] Unit test for format detection in tests/unit/test_format_detector.cpp
- [ ] T040 [P] [US1] Integration test for DSD1024 file processing in tests/integration/test_dsd_processing.cpp
- [ ] T041 [P] [US1] Performance test for file processing latency in tests/performance/test_file_processing.cpp

### Audio Format Support Implementation
- [ ] T042 [P] [US1] Create MP3 format decoder in src/core/fileio/decoders/mp3_decoder.cpp/hpp
- [ ] T043 [P] [US1] Create WAV format decoder in src/core/fileio/decoders/wav_decoder.cpp/hpp
- [ ] T044 [P] [US1] Create FLAC format decoder in src/core/fileio/decoders/flac_decoder.cpp/hpp
- [ ] T045 [P] [US1] Create ALAC format decoder in src/core/fileio/decoders/alac_decoder.cpp/hpp
- [ ] T046 [P] [US1] Create AAC format decoder in src/core/fileio/decoders/aac_decoder.cpp/hpp
- [ ] T047 [P] [US1] Create OGG format decoder in src/core/fileio/decoders/ogg_decoder.cpp/hpp
- [ ] T048 [P] [US1] Create M4A format decoder in src/core/fileio/decoders/m4a_decoder.cpp/hpp

### High-Resolution Audio Support
- [ ] T049 [P] [US1] Create DSD64 decoder in src/core/fileio/decoders/dsd64_decoder.cpp/hpp
- [ ] T050 [P] [US1] Create DSD128 decoder in src/core/fileio/decoders/dsd128_decoder.cpp/hpp
- [ ] T051 [P] [US1] Create DSD256 decoder in src/core/fileio/decoders/dsd256_decoder.cpp/hpp
- [ ] T052 [P] [US1] Create DSD512 decoder in src/core/fileio/decoders/dsd512_decoder.cpp/hpp
- [ ] T053 [US1] Create DSD1024 decoder in src/core/fileio/decoders/dsd1024_decoder.cpp/hpp
- [ ] T054 [US1] Create DSF format decoder in src/core/fileio/decoders/dsf_decoder.cpp/hpp
- [ ] T055 [P] [US1] Create DFF format decoder in src/core/fileio/decoders/dff_decoder.cpp/hpp

### API Implementation
- [ ] T056 [US1] Implement audio file upload endpoint in src/network/http_endpoints/audio_upload.cpp/hpp
- [ ] T057 [US1] Implement audio formats endpoint in src/network/http_endpoints/audio_formats.cpp/hpp
- [ ] T058 [US1] Implement audio metadata endpoint in src/network/http_endpoints/audio_metadata.cpp/hpp
- [ ] T059 [US1] Implement audio status endpoint in src/network/http_endpoints/audio_status.cpp/hpp

### Core Processing Implementation
- [ ] T060 [US1] Implement DSD1024 processor in src/core/dsp/dsd_processor.cpp/hpp (depends on T017 GPU processor)
- [ ] T061 [US1] Create audio processing pipeline in src/core/audio_pipeline.cpp/hpp
- [ ] T062 [US1] Implement processing progress tracking in src/core/processing_progress.cpp/hpp

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Real-time Audio Visualization (Priority: P1)

**Goal**: Provide real-time spectrum analysis, waveform display, VU meters, and hardware monitoring

**Independent Test**: Can be fully tested by playing audio files and verifying all visualization components update smoothly with accurate data representation

### Tests for User Story 2 (TDD Required) ⚠️

- [ ] T063 [P] [US2] Contract test for WebSocket real-time data in tests/contract/test_websocket_realtime.cpp
- [ ] T064 [P] [US2] Unit test for spectrum analysis in tests/unit/test_spectrum_analyzer.cpp
- [ ] T065 [P] [US2] Performance test for 60fps visualization in tests/performance/test_visualization_fps.cpp

### Real-time Data Processing
- [ ] T066 [P] [US2] Implement spectrum analyzer with GPU acceleration in src/core/dsp/spectrum_analyzer.cpp/hpp
- [ ] T067 [P] [US2] Create waveform processor in src/core/dsp/waveform_processor.cpp/hpp
- [ ] T068 [P] [US2] Implement VU meter processor in src/core/dsp/vu_meter.cpp/hpp

### GPU Shaders for Visualization
- [ ] T069 [P] [US2] Create spectrum analysis compute shader in shaders/spectrum_analyzer.comp
- [ ] T070 [P] [US2] Create waveform analysis compute shader in shaders/waveform_analyzer.comp

### WebSocket Real-time Implementation
- [ ] T071 [US2] Implement real-time WebSocket data streaming in src/network/realtime_websocket.cpp/hpp
- [ ] T072 [US2] Create binary protocol for audio visualization data in src/network/protocol/visualization_protocol.cpp/hpp
- [ ] T073 [US2] Implement WebSocket client subscription management in src/network/subscription_manager.cpp/hpp

### Hardware Monitoring Integration
- [ ] T074 [US2] Enhance hardware monitor for real-time data in src/system/hardware_monitor.cpp/hpp
- [ ] T075 [US2] Implement GPU utilization real-time tracking in src/system/gpu_monitor.cpp/hpp
- [ ] T076 [US2] Create processing metrics collector in src/system/processing_metrics.cpp/hpp

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Multi-Device Audio Output (Priority: P2)

**Goal**: Enable discovery and seamless switching between local speakers, Roon Bridge, HQPlayer NAA, and UPnP renderers

**Independent Test**: Can be fully tested by discovering available devices on the network and verifying audio routing to each target device

### Tests for User Story 3 (TDD Required) ⚠️

- [ ] T077 [P] [US3] Contract test for device discovery in tests/contract/test_device_discovery.cpp
- [ ] T078 [P] [US3] Integration test for Roon Bridge integration in tests/integration/test_roon_bridge.cpp
- [ ] T079 [P] [US3] Integration test for HQPlayer NAA integration in tests/integration/test_hqplayer_naa.cpp
- [ ] T080 [P] [US3] Integration test for UPnP renderer integration in tests/integration/test_upnp_renderer.cpp

### Device Discovery Infrastructure
- [ ] T081 [P] [US3] Create device discovery service in src/network/discovery_service.cpp/hpp
- [ ] T082 [P] [US3] Implement mDNS discovery for Roon Bridge in src/network/md_discovery.cpp/hpp
- [ ] T083 [P] [US3] Implement UPnP SSDP discovery in src/network/upnp_discovery.cpp/hpp
- [ ] T084 [P] [US3] Create network scanner for HQPlayer NAA in src/network/nethardware_scanner.cpp/hpp

### Output Device Management
- [ ] T085 [US3] Create output manager in src/output/output_manager.cpp/hpp
- [ ] T086 [P] [US3] Implement local audio output in src/output/local_output.cpp/hpp
- [ ] T087 [US3] Create output device abstract base class in src/output/audio_device.cpp/hpp

### Roon Bridge Integration
- [ ] T088 [US3] Implement Roon Bridge client in src/output/roon_bridge.cpp/hpp
- [ ] T089 [US3] Create Roon protocol handler in src/output/roon_protocol.cpp/hpp
- [ ] T090 [P] [US3] Implement Roon metadata synchronization in src/output/roon_metadata.cpp/hpp

### HQPlayer NAA Integration
- [ ] T091 [US3] Implement HQPlayer NAA client in src/output/hqplayer_naa.cpp/hpp
- [ ] T092 [P] [US3] Create NAA protocol handler in src/output/naa_protocol.cpp/hpp
- [ ] T093 [P] [US3] Implement HQPlayer format negotiation in src/output/hqp_format_negotiation.cpp/hpp

### UPnP Renderer Integration
- [ ] T094 [US3] Implement UPnP renderer client in src/output/upnp_renderer.cpp/hpp
- [ ] T095 [P] [US3] Create UPnP media renderer protocol in src/output/upnp_protocol.cpp/hpp
- [ ] T096 [P] [US3] Implement UPnP device capability detection in src/output/upnp_capabilities.cpp/hpp

### API Implementation
- [ ] T097 [US3] Implement device discovery endpoint in src/network/http_endpoints/output_discover.cpp/hpp
- [ ] T098 [US3] Implement device selection endpoint in src/network/http_endpoints/output_select.cpp/hpp
- [ ] T099 [US3] Implement device status endpoint in src/network/http_endpoints/output_status.cpp/hpp

**Checkpoint**: All user stories 1-3 should now be independently functional

---

## Phase 6: User Story 4 - Audio Processing Chain Management (Priority: P2)

**Goal**: Enable creation and management of processing chains with 512-band EQ, 16M-point convolution, and custom effects

**Independent Test**: Can be fully tested by adding filters to the chain, adjusting parameters, and verifying audio output changes accordingly

### Tests for User Story 4 (TDD Required) ⚠️

- [ ] T100 [P] [US4] Contract test for processing chain management in tests/contract/test_processing_chain.cpp
- [ ] T101 [P] [US4] Unit test for 512-band EQ in tests/unit/test_equalizer.cpp
- [ ] T102 [P] [US4] Performance test for 16M-point convolution in tests/performance/test_convolution.cpp
- [ ] T103 [P] [US4] GPU test for filter chain processing in tests/gpu/test_filter_chain_gpu.cpp

### Filter Infrastructure
- [ ] T104 [P] [US4] Create filter base class in src/core/filters/filter_base.cpp/hpp
- [ ] T105 [P] [US4] Implement filter chain manager in src/core/filters/filter_chain.cpp/hpp
- [ ] T106 [P] [US4] Create filter parameter manager in src/core/filters/parameter_manager.cpp/hpp

### 512-Band Equalizer Implementation
- [ ] T107 [US4] Implement 512-band EQ processor in src/core/dsp/eq_processor.cpp/hpp
- [ ] T108 [P] [US4] Create EQ filter bank in src/core/filters/eq_filter_bank.cpp/hpp
- [ ] T109 [P] [US4] Implement frequency-domain EQ processing in src/core/filters/frequency_eq.cpp/hpp
- [ ] T110 [P] [US4] Create EQ coefficient calculator in src/core/filters/eq_coefficients.cpp/hpp

### 16M-Point Convolution Implementation
- [ ] T111 [US4] Implement 16M-point convolver in src/core/dsp/convolver.cpp/hpp
- [ ] T112 [P] [US4] Create partitioned convolution processor in src/core/filters/partitioned_convolution.cpp/hpp
- [ ] T113 [P] [US4] Implement impulse response loader in src/core/filters/ir_loader.cpp/hpp
- [ ] T114 [US4] Create convolution GPU shader in shaders/convolution.comp
- [ ] T115 [P] [US4] Implement FFT-based convolution in src/core/filters/fft_convolution.cpp/hpp

### Additional Filters
- [ ] T116 [P] [US4] Create biquad filter implementation in src/core/filters/biquad_filter.cpp/hpp
- [ ] T117 [P] [US4] Implement FIR filter in src/core/filters/fir_filter.cpp/hpp
- [ ] T118 [P] [US4] Create resampler for format conversion in src/core/dsp/resampler.cpp/hpp

### Processing Chain API
- [ ] T119 [US4] Implement filter list endpoint in src/network/http_endpoints/filters.cpp/hpp
- [ ] T120 [US4] Implement chain management endpoint in src/network/http_endpoints/chain.cpp/hpp
- [ ] T121 [US4] Implement filter parameter endpoint in src/network/http_endpoints/filter_parameters.cpp/hpp
- [ ] T122 [US4] Implement filter control endpoint in src/network/http_endpoints/filter_control.cpp/hpp

**Checkpoint**: All user stories should now be independently functional

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

### Performance Optimization
- [ ] T123 [P] Optimize GPU memory usage for concurrent processing in src/core/gpu/memory_optimizer.cpp/hpp
- [ ] T124 [P] Implement adaptive quality scaling in src/network/adaptive_quality.cpp/hpp
- [ ] T125 [P] Optimize WebSocket binary protocol bandwidth in src/network/protocol/binary_optimizer.cpp/hpp

### Error Handling & Recovery
- [ ] T126 [P] Implement graceful GPU fallback in src/core/gpu/gpu_fallback.cpp/hpp
- [ ] T127 [P] Create audio buffer underrun recovery in src/core/audio_recovery.cpp/hpp
- [ ] T128 [P] Implement network reconnection logic in src/network/connection_recovery.cpp/hpp

### Documentation & Deployment
- [ ] T129 [P] Create comprehensive API documentation in docs/api/
- [ ] T130 [P] Update README.md with installation and usage instructions
- [ ] T131 [P] Create performance benchmarks in tools/benchmark/
- [ ] T132 [P] Add deployment configuration for production environments in config/production.json

### Additional Testing
- [ ] T133 [P] Create end-to-end integration tests in tests/e2e/
- [ ] T134 [P] Implement 24-hour stress test in tests/stress/test_24hour_stability.cpp
- [ ] T135 [P] Create multi-user concurrent processing test in tests/integration/test_concurrent_users.cpp

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (US1 → US2 → US3 → US4)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable
- **User Story 4 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1/US2/US3 but should be independently testable

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models/Entities before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together (if tests requested):
Task: "Contract test for audio upload endpoint in tests/contract/test_audio_upload.cpp"
Task: "Unit test for format detection in tests/unit/test_format_detector.cpp"
Task: "Integration test for DSD1024 file processing in tests/integration/test_dsd_processing.cpp"
Task: "Performance test for file processing latency in tests/performance/test_file_processing.cpp"

# Launch all format decoders in parallel:
Task: "Create MP3 format decoder in src/core/fileio/decoders/mp3_decoder.cpp/hpp"
Task: "Create WAV format decoder in src/core/fileio/decoders/wav_decoder.cpp/hpp"
Task: "Create FLAC format decoder in src/core/fileio/decoders/flac_decoder.cpp/hpp"
Task: "Create ALAC format decoder in src/core/fileio/decoders/alac_decoder.cpp/hpp"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Add User Story 4 → Test independently → Deploy/Demo
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (Audio File Processing)
   - Developer B: User Story 2 (Real-time Visualization)
   - Developer C: User Story 3 (Multi-Device Output)
   - Developer D: User Story 4 (Processing Chain)
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence

**Total Tasks**: 135 implementation tasks
**Tasks per User Story**: US1 (38), US2 (14), US3 (26), US4 (19), Setup/Foundational (33), Polish (15)
**Parallel Opportunities**: 85 tasks marked as parallelizable
**Independent Test Criteria**: Each user story can be tested and demonstrated independently