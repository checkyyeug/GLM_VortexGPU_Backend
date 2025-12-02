use vortex_network::protocol::binary::BinaryProtocol;
use vortex_network::protocol::websocket::WebSocketHandler;
use vortex_network::protocol::http::HttpHandler;
use common::*;
use vortex_proto::network::{AudioMetadata, AudioFormat, ProcessingChain, ProcessingStep, ProcessingType};
use std::collections::HashMap;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

mod common;

#[tokio::test]
async fn test_binary_protocol_serialization() {
    init_test_logger();

    let metadata = create_test_audio_metadata();
    let proto_msg = network_message::Payload::AudioMetadata(metadata);

    // Test serialization
    let serialized = BinaryProtocol::serialize_message(&proto_msg)
        .expect("Failed to serialize message");

    assert!(!serialized.is_empty());
    assert!(serialized.len() > 8); // Header + data

    // Test deserialization
    let deserialized = BinaryProtocol::deserialize_message(&serialized)
        .expect("Failed to deserialize message");

    match deserialized.payload {
        Some(network_message::Payload::AudioMetadata(deserialized_metadata)) => {
            assert_eq!(deserialized_metadata.title, "Test Track");
            assert_eq!(deserialized_metadata.artist, "Test Artist");
            assert_eq!(deserialized_metadata.format, AudioFormat::Pcm as i32);
            assert_eq!(deserialized_metadata.sample_rate, 44100);
        }
        _ => panic!("Expected AudioMetadata payload"),
    }
}

#[tokio::test]
async fn test_processing_chain_validation() {
    init_test_logger();

    let valid_chain = ProcessingChain {
        id: "test-chain".to_string(),
        name: "Test Processing Chain".to_string(),
        steps: vec![
            ProcessingStep {
                id: "eq-1".to_string(),
                step_type: ProcessingType::Equalizer as i32,
                parameters: HashMap::from([
                    ("frequencies".to_string(), "[100,1000,10000]".to_string()),
                    ("gains".to_string(), "[0.0,2.0,-1.0]".to_string()),
                    ("q".to_string(), "[1.0,1.414,1.0]".to_string()),
                ]),
                enabled: true,
                order: 0,
            },
            ProcessingStep {
                id: "conv-1".to_string(),
                step_type: ProcessingType::Convolution as i32,
                parameters: HashMap::from([
                    ("impulse_file".to_string(), "/test/impulse.wav".to_string()),
                    ("length".to_string(), "1048576".to_string()),
                ]),
                enabled: true,
                order: 1,
            },
        ],
        enabled: true,
    };

    // Test valid chain
    assert!(BinaryProtocol::validate_processing_chain(&valid_chain).is_ok());

    // Test invalid chain (missing parameters)
    let mut invalid_chain = valid_chain.clone();
    invalid_chain.steps[0].parameters.remove("frequencies");
    assert!(BinaryProtocol::validate_processing_chain(&invalid_chain).is_err());
}

#[tokio::test]
async fn test_websocket_message_format() {
    init_test_logger();

    let message = create_test_websocket_message();

    // Test JSON serialization
    let json = serde_json::to_string(&message)
        .expect("Failed to serialize WebSocket message");

    assert!(json.contains("\"channel\":\"visualization\""));
    assert!(json.contains("\"type\":1")); // MessageType::Data

    // Test JSON deserialization
    let deserialized: WebSocketMessage = serde_json::from_str(&json)
        .expect("Failed to deserialize WebSocket message");

    assert_eq!(deserialized.channel, "visualization");
    assert_eq!(deserialized.message_type, MessageType::Data as i32);
}

#[tokio::test]
async fn test_audio_buffer_operations() {
    init_test_logger();

    let test_data = TestDataBuilder::new()
        .with_sine_wave(44100, 1000.0, Duration::from_millis(100))
        .build();

    // Test buffer alignment
    assert_eq!(test_data.len() % 8, 0, "Buffer should be 8-byte aligned");

    // Test buffer conversion
    let samples: Vec<f32> = test_data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    assert!(!samples.is_empty());
    assert_eq!(samples.len() * 4, test_data.len());

    // Test RMS calculation
    let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
    assert!(rms > 0.0, "Sine wave should have non-zero RMS");
}

#[tokio::test]
async fn test_network_chunk_handling() {
    init_test_logger();

    let large_data = TestDataBuilder::new()
        .with_sine_wave(44100, 440.0, Duration::from_millis(1000))
        .build();

    let chunks = BinaryProtocol::create_chunks(&large_data, 4096);

    assert!(!chunks.is_empty());
    assert!(chunks.len() > 1, "Large data should be split into multiple chunks");

    let mut reconstructed = Vec::new();
    for chunk in chunks {
        reconstructed.extend_from_slice(&chunk.data);
    }

    assert_eq!(reconstructed.len(), large_data.len());
    assert_eq!(reconstructed, large_data);
}

#[tokio::test]
async fn test_http_request_validation() {
    init_test_logger();

    // Test valid upload request
    let valid_metadata = create_test_audio_metadata();
    assert!(HttpHandler::validate_audio_metadata(&valid_metadata).is_ok());

    // Test invalid metadata (missing required fields)
    let mut invalid_metadata = valid_metadata.clone();
    invalid_metadata.sample_rate = 0;
    assert!(HttpHandler::validate_audio_metadata(&invalid_metadata).is_err());

    invalid_metadata = valid_metadata.clone();
    invalid_metadata.channels = 0;
    assert!(HttpHandler::validate_audio_metadata(&invalid_metadata).is_err());

    invalid_metadata = valid_metadata.clone();
    invalid_metadata.bit_depth = 0;
    assert!(HttpHandler::validate_audio_metadata(&invalid_metadata).is_err());
}

#[tokio::test]
async fn test_concurrent_message_handling() {
    init_test_logger();

    let handler = WebSocketHandler::new();

    let tasks: Vec<_> = (0..10).map(|i| {
        let handler = handler.clone();
        tokio::spawn(async move {
            let message = WebSocketMessage {
                message_type: MessageType::Data as i32,
                channel: format!("test_channel_{}", i),
                timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                payload: Some(websocket_message::Payload::Heartbeat(())),
            };

            handler.process_message(message).await
        })
    }).collect();

    // Wait for all tasks to complete
    for task in tasks {
        let result = task.await.expect("Task panicked");
        assert!(result.is_ok(), "Concurrent message processing should succeed");
    }
}

#[tokio::test]
async fn test_error_handling() {
    init_test_logger();

    // Test malformed message
    let malformed_data = vec![0xFF; 100]; // Invalid binary data
    let result = BinaryProtocol::deserialize_message(&malformed_data);
    assert!(result.is_err());

    // Test chunk reconstruction errors
    let corrupted_chunks = vec![
        NetworkChunk {
            chunk_id: 1,
            total_chunks: 2,
            sequence_number: 1,
            data: vec![1, 2, 3],
            checksum: vec![0, 0, 0, 0], // Invalid checksum
        },
    ];

    let result = BinaryProtocol::reconstruct_chunks(corrupted_chunks);
    assert!(result.is_err());
}

realtime_test!(test_realtime_serialization, 1, {
    let metadata = create_test_audio_metadata();
    let proto_msg = network_message::Payload::AudioMetadata(metadata);

    let _serialized = BinaryProtocol::serialize_message(&proto_msg)
        .expect("Failed to serialize message");

    let _deserialized = BinaryProtocol::deserialize_message(&_serialized)
        .expect("Failed to deserialize message");
});

#[tokio::test]
async fn test_memory_efficiency() {
    init_test_logger();

    let large_metadata = AudioMetadata {
        title: "A".repeat(1000),
        artist: "B".repeat(1000),
        album: "C".repeat(1000),
        ..create_test_audio_metadata()
    };

    let serialized = BinaryProtocol::serialize_message(&network_message::Payload::AudioMetadata(large_metadata))
        .expect("Failed to serialize message");

    // Ensure compressed size is reasonable (less than 50% of original)
    let original_size = 3000; // Approximate original string length
    let compression_ratio = serialized.len() as f64 / original_size as f64;
    assert!(compression_ratio < 0.5, "Compression ratio should be < 0.5, got {}", compression_ratio);
}