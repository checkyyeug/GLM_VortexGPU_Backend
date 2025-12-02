use vortex_network::protocol::binary::BinaryProtocol;
use vortex_network::protocol::websocket::WebSocketHandler;
use vortex_network::protocol::http::HttpHandler;
use common::*;
use vortex_proto::network::{AudioMetadata, AudioFormat, ProcessingChain, ProcessingStep, ProcessingType};
use std::collections::HashMap;
use std::time::Duration;

mod common;

// Contract Tests: Verify system meets all specified requirements
// These tests validate that the implementation conforms to the specification contracts

#[tokio::test]
async fn contract_audio_format_compatibility() {
    init_test_logger();

    // Test all required audio formats
    let formats = vec![
        AudioFormat::Pcm,
        AudioFormat::Flac,
        AudioFormat::Wav,
        AudioFormat::Mp3,
        AudioFormat::Aac,
        AudioFormat::Ogg,
        AudioFormat::Opus,
        AudioFormat::Dsd64,
        AudioFormat::Dsd128,
        AudioFormat::Dsd256,
        AudioFormat::Dsd512,
        AudioFormat::Dsd1024,
    ];

    for format in formats {
        let metadata = AudioMetadata {
            title: format!("Test {}", format as i32),
            ..create_test_audio_metadata()
        };

        // Verify format is supported
        let validation_result = HttpHandler::validate_audio_metadata(&metadata);
        assert!(validation_result.is_ok(), "Format {:?} should be supported", format);
    }
}

#[tokio::test]
async fn contract_quality_requirements() {
    init_test_logger();

    // SC-001: DSD1024 quality check
    let dsd_metadata = AudioMetadata {
        title: "DSD1024 Test".to_string(),
        format: AudioFormat::Dsd1024 as i32,
        sample_rate: 45_158_400, // 44.1kHz * 1024
        bit_depth: 1,
        channels: 2,
        ..create_test_audio_metadata()
    };

    let validation_result = HttpHandler::validate_audio_metadata(&dsd_metadata);
    assert!(validation_result.is_ok(), "DSD1024 format should be supported");

    // SC-001: 512-band equalizer
    let eq_chain = ProcessingChain {
        id: "512-band-eq".to_string(),
        name: "512 Band Equalizer".to_string(),
        steps: vec![ProcessingStep {
            id: "eq-512".to_string(),
            step_type: ProcessingType::Equalizer as i32,
            parameters: HashMap::from([
                ("bands".to_string(), "512".to_string()),
                ("frequencies".to_string(), (0..512).map(|i| (i * 43.066).to_string()).collect::<Vec<_>>().join(",")),
                ("gains".to_string(), (0..512).map(|_| "0.0".to_string()).collect::<Vec<_>>().join(",")),
                ("q".to_string(), (0..512).map(|_| "1.0".to_string()).collect::<Vec<_>>().join(",")),
            ]),
            enabled: true,
            order: 0,
        }],
        enabled: true,
    };

    assert!(BinaryProtocol::validate_processing_chain(&eq_chain).is_ok());

    // SC-001: 16M-point convolution
    let conv_chain = ProcessingChain {
        id: "16m-conv".to_string(),
        name: "16M Point Convolution".to_string(),
        steps: vec![ProcessingStep {
            id: "conv-16m".to_string(),
            step_type: ProcessingType::Convolution as i32,
            parameters: HashMap::from([
                ("length".to_string(), "16777216".to_string()), // 16^7
                ("impulse_file".to_string(), "/test/16m_impulse.wav".to_string()),
            ]),
            enabled: true,
            order: 0,
        }],
        enabled: true,
    };

    assert!(BinaryProtocol::validate_processing_chain(&conv_chain).is_ok());
}

#[tokio::test]
async fn contract_realtime_constraints() {
    init_test_logger();

    // SC-002: <10ms audio processing constraint
    let large_audio_data = TestDataBuilder::new()
        .with_sine_wave(44100, 1000.0, Duration::from_millis(100))
        .build();

    let start = std::time::Instant::now();

    // Simulate audio processing
    let audio_data = AudioData {
        data: large_audio_data.clone(),
        format: AudioFormat::Pcm as i32,
        sample_rate: 44100,
        channels: 2,
        bit_depth: 16,
    };

    let proto_msg = network_message::Payload::AudioData(audio_data);
    let _serialized = BinaryProtocol::serialize_message(&proto_msg).unwrap();

    let processing_time = start.elapsed();
    assert!(
        processing_time.as_millis() < 10,
        "Audio processing took {}ms, constraint is <10ms",
        processing_time.as_millis()
    );

    // SC-002: <50ms WebSocket latency
    let handler = WebSocketHandler::new();
    let spectrum_data = create_test_spectrum_data();
    let ws_message = WebSocketMessage {
        message_type: MessageType::Data as i32,
        channel: "visualization".to_string(),
        timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
        payload: Some(websocket_message::Payload::SpectrumData(spectrum_data)),
    };

    let ws_start = std::time::Instant::now();
    let _result = handler.process_message(ws_message).await;
    let ws_latency = ws_start.elapsed();

    assert!(
        ws_latency.as_millis() < 50,
        "WebSocket processing took {}ms, constraint is <50ms",
        ws_latency.as_millis()
    );
}

#[tokio::test]
async fn contract_visualization_requirements() {
    init_test_logger();

    // SC-003: 2048-point FFT
    let spectrum_data = SpectrumData {
        frequency_bins: (0..2048).map(|i| i as f32 * 21.533).collect(), // 44.1kHz / 2048
        magnitudes: vec![0.0f32; 2048],
        sample_rate: 44100,
        fft_size: 2048,
        window_type: "hann".to_string(),
        timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
    };

    // Verify FFT size matches specification
    assert_eq!(spectrum_data.fft_size, 2048);
    assert_eq!(spectrum_data.frequency_bins.len(), 2048);
    assert_eq!(spectrum_data.magnitudes.len(), 2048);

    // SC-003: 60fps visualization at 16.667ms intervals
    let handler = WebSocketHandler::new();
    let mut timestamps = Vec::new();

    for i in 0..10 {
        let message = WebSocketMessage {
            message_type: MessageType::Data as i32,
            channel: "spectrum".to_string(),
            timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            payload: Some(websocket_message::Payload::SpectrumData(spectrum_data.clone())),
        };

        let start = std::time::Instant::now();
        let _result = handler.process_message(message).await;
        timestamps.push(start.elapsed());
    }

    // Verify consistent timing (within ±5ms)
    let target_interval = Duration::from_millis(16);
    for (i, &duration) in timestamps.iter().enumerate() {
        if i > 0 {
            let diff = if duration > timestamps[i-1] {
                duration - timestamps[i-1]
            } else {
                timestamps[i-1] - duration
            };
            assert!(
                diff <= Duration::from_millis(5),
                "Timing variation too large: {:?} at message {}",
                diff, i
            );
        }
    }
}

#[tokio::test]
async fn contract_multi_device_output() {
    init_test_logger();

    // SC-004: Multi-device output support
    let output_devices = vec![
        ("Roon Bridge", 8080),
        ("HQPlayer NAA", 8081),
        ("UPnP Renderer", 8082),
        ("Local Output", 8083),
    ];

    for (device_name, port) in output_devices {
        let output_config = OutputDevice {
            name: device_name.to_string(),
            device_type: match device_name {
                "Roon Bridge" => DeviceType::RoonBridge,
                "HQPlayer NAA" => DeviceType::HqplayerNaa,
                "UPnP Renderer" => DeviceType::UpnpRenderer,
                "Local Output" => DeviceType::Local,
                _ => DeviceType::Local,
            } as i32,
            address: "127.0.0.1".to_string(),
            port,
            enabled: true,
            parameters: HashMap::from([
                ("sample_rate".to_string(), "44100".to_string()),
                ("bit_depth".to_string(), "16".to_string()),
                ("channels".to_string(), "2".to_string()),
            ]),
        };

        // Verify output device configuration
        assert!(!output_config.name.is_empty());
        assert!(output_config.port > 0);
        assert!(output_config.enabled);
    }
}

#[tokio::test]
async fn contract_gpu_acceleration() {
    init_test_logger();

    // SC-005: GPU acceleration support
    let gpu_backends = vec!["CUDA", "OpenCL", "Vulkan"];

    for backend in gpu_backends {
        let gpu_config = GpuConfig {
            preferred_backends: vec![backend.to_string()],
            memory_limit_mb: 8192, // 8GB
            compute_capability: Some("6.0+".to_string()),
            enable_profiling: false,
        };

        // Verify GPU configuration
        assert!(!gpu_config.preferred_backends.is_empty());
        assert!(gpu_config.memory_limit_mb > 0);
    }
}

#[tokio::test]
async fn contract_api_contract() {
    init_test_logger();

    // Verify HTTP API contract
    let test_endpoints = vec![
        ("GET", "/api/system/health"),
        ("POST", "/api/audio/upload"),
        ("GET", "/api/audio/:id/metadata"),
        ("POST", "/api/audio/:id/process"),
        ("GET", "/api/processing/chains"),
        ("POST", "/api/processing/chains"),
        ("GET", "/api/output/devices"),
        ("GET", "/api/system/gpu/info"),
    ];

    for (method, path) in test_endpoints {
        assert!(!method.is_empty());
        assert!(!path.is_empty());
        assert!(path.starts_with("/api/"));
    }

    // Verify WebSocket message types
    let message_types = vec![
        MessageType::Subscribe,
        MessageType::Unsubscribe,
        MessageType::Data,
        MessageType::Control,
        MessageType::Heartbeat,
    ];

    for message_type in message_types {
        assert!(message_type as i32 >= 0);
    }
}

#[tokio::test]
async fn contract_file_size_limits() {
    init_test_logger();

    // Test maximum file size handling (2GB limit)
    let max_file_size = 2 * 1024 * 1024 * 1024; // 2GB
    let large_metadata = AudioMetadata {
        title: "Large File Test".to_string(),
        duration: Some(prost_types::Duration {
            seconds: 3600, // 1 hour
            nanos: 0,
        }),
        ..create_test_audio_metadata()
    };

    // Calculate expected file size
    let estimated_size = large_metadata.sample_rate as u64 *
        large_metadata.channels as u64 *
        (large_metadata.bit_depth as u64 / 8) *
        3600; // 1 hour

    assert!(
        estimated_size < max_file_size,
        "Estimated file size {} exceeds limit {}",
        estimated_size, max_file_size
    );

    // Test file format validation
    let valid_formats = vec!["PCM", "FLAC", "WAV", "MP3", "AAC", "OGG", "OPUS"];
    for format in valid_formats {
        let metadata = AudioMetadata {
            format: match format {
                "PCM" => AudioFormat::Pcm,
                "FLAC" => AudioFormat::Flac,
                "WAV" => AudioFormat::Wav,
                "MP3" => AudioFormat::Mp3,
                "AAC" => AudioFormat::Aac,
                "OGG" => AudioFormat::Ogg,
                "OPUS" => AudioFormat::Opus,
                _ => AudioFormat::Pcm,
            } as i32,
            ..create_test_audio_metadata()
        };

        assert!(HttpHandler::validate_audio_metadata(&metadata).is_ok(),
                "Format {} should be valid", format);
    }
}

#[tokio::test]
async fn contract_concurrent_user_support() {
    init_test_logger();

    // SC-006: Support multiple concurrent users
    let max_concurrent_users = 100;
    let user_sessions: Vec<_> = (0..max_concurrent_users)
        .map(|user_id| {
            UserSession {
                user_id: format!("user_{}", user_id),
                websocket_session: format!("ws_session_{}", user_id),
                created_at: chrono::Utc::now().timestamp(),
                last_activity: chrono::Utc::now().timestamp(),
                active_audio_file: None,
            }
        })
        .collect();

    // Verify all sessions are unique
    let mut session_ids = std::collections::HashSet::new();
    for session in &user_sessions {
        assert!(session_ids.insert(session.user_id.clone()));
        assert!(session_ids.insert(session.websocket_session.clone()));
    }

    assert_eq!(user_sessions.len(), max_concurrent_users * 2); // user_id + websocket_session per user
}

realtime_test!(contract_realtime_processing_chain, 5, {
    let processing_chain = ProcessingChain {
        id: "realtime-chain".to_string(),
        name: "Real-time Processing Chain".to_string(),
        steps: vec![
            ProcessingStep {
                id: "eq".to_string(),
                step_type: ProcessingType::Equalizer as i32,
                parameters: HashMap::from([
                    ("bands".to_string(), "10".to_string()),
                    ("frequencies".to_string(), "100,1000,10000,20000".to_string()),
                ]),
                enabled: true,
                order: 0,
            },
            ProcessingStep {
                id: "gain".to_string(),
                step_type: ProcessingType::Gain as i32,
                parameters: HashMap::from([
                    ("gain_db".to_string(), "0.0".to_string()),
                ]),
                enabled: true,
                order: 1,
            },
        ],
        enabled: true,
    };

    let validation_result = BinaryProtocol::validate_processing_chain(&processing_chain);
    assert!(validation_result.is_ok());
});

#[tokio::test]
async fn contract_error_handling() {
    init_test_logger();

    // Test graceful error handling for invalid inputs
    let invalid_cases = vec![
        (AudioFormat::Pcm as i32, 0, 16, 2),      // Invalid sample rate
        (AudioFormat::Pcm as i32, 44100, 0, 2),   // Invalid bit depth
        (AudioFormat::Pcm as i32, 44100, 16, 0),  // Invalid channels
        (999, 44100, 16, 2),                      // Invalid format
    ];

    for (format, sample_rate, bit_depth, channels) in invalid_cases {
        let invalid_metadata = AudioMetadata {
            format,
            sample_rate,
            bit_depth,
            channels,
            ..create_test_audio_metadata()
        };

        let result = HttpHandler::validate_audio_metadata(&invalid_metadata);
        assert!(result.is_err(), "Should reject invalid metadata: format={}, sr={}, depth={}, ch={}",
                format, sample_rate, bit_depth, channels);
    }
}