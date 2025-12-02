use vortex_network::protocol::binary::BinaryProtocol;
use vortex_network::protocol::websocket::WebSocketHandler;
use vortex_network::protocol::http::HttpHandler;
use vortex_network::discovery::DiscoveryService;
use common::*;
use vortex_proto::network::{AudioMetadata, AudioFormat, ProcessingChain, ProcessingStep, ProcessingType};
use std::collections::HashMap;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{SinkExt, StreamExt};

mod common;

#[tokio::test]
async fn test_end_to_end_audio_upload() {
    init_test_logger();

    // Setup test server
    let listener = setup_test_server().await;
    let port = listener.local_addr().unwrap().port();

    // Start HTTP handler in background
    let http_handler = HttpHandler::new();
    let handler_clone = http_handler.clone();
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            let handler = handler_clone.clone();
            tokio::spawn(async move {
                handler.handle_connection(stream).await;
            });
        }
    });

    // Wait for server to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Test file upload
    let client = reqwest::Client::new();
    let test_data = TestDataBuilder::new()
        .with_sine_wave(44100, 1000.0, Duration::from_millis(100))
        .build();

    let metadata = create_test_audio_metadata();
    let metadata_json = serde_json::to_string(&metadata).unwrap();

    let response = client
        .post(&format!("http://127.0.0.1:{}/api/audio/upload", port))
        .header("X-Audio-Metadata", metadata_json)
        .header("Content-Type", "application/octet-stream")
        .body(test_data.clone())
        .send()
        .await
        .expect("Failed to upload audio file");

    assert_eq!(response.status(), 200);

    let upload_result: serde_json::Value = response.json().await.unwrap();
    assert!(upload_result["file_id"].is_string());
    assert!(upload_result["status"] == "success");
}

#[tokio::test]
async fn test_websocket_realtime_data() {
    init_test_logger();

    let websocket_port = 8081;

    // Start WebSocket handler
    let ws_handler = WebSocketHandler::new();
    let handler_clone = ws_handler.clone();
    tokio::spawn(async move {
        handler_clone.start(websocket_port).await;
    });

    // Wait for server to start
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Connect WebSocket client
    let (ws_stream, _) = connect_async(&format!("ws://127.0.0.1:{}", websocket_port))
        .await
        .expect("Failed to connect to WebSocket");

    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    // Subscribe to visualization data
    let subscribe_msg = WebSocketMessage {
        message_type: MessageType::Subscribe as i32,
        channel: "visualization".to_string(),
        timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
        payload: Some(websocket_message::Payload::SubscribeRequest(
            websocket_message::SubscribeRequest {
                channel: "visualization".to_string(),
                subscription_type: "spectrum".to_string(),
                frequency: 60.0,
                parameters: HashMap::new(),
            }
        )),
    };

    let subscribe_json = serde_json::to_string(&subscribe_msg).unwrap();
    ws_sender.send(Message::Text(subscribe_json)).await.unwrap();

    // Wait for subscription confirmation
    let timeout = Duration::from_secs(5);
    let start = std::time::Instant::now();

    while start.elapsed() < timeout {
        if let Some(msg) = ws_receiver.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    let ws_msg: WebSocketMessage = serde_json::from_str(&text).unwrap();
                    if ws_msg.message_type == MessageType::SubscriptionAck as i32 {
                        break;
                    }
                }
                Ok(Message::Close(_)) => panic!("WebSocket closed unexpectedly"),
                Err(e) => panic!("WebSocket error: {}", e),
                _ => continue,
            }
        }
    }

    // Test real-time data reception
    let messages_received = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let messages_clone = messages_received.clone();

    let data_task = tokio::spawn(async move {
        let mut count = 0;
        let start_time = std::time::Instant::now();

        while start_time.elapsed() < Duration::from_millis(500) {
            if let Some(msg) = ws_receiver.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        let ws_msg: WebSocketMessage = serde_json::from_str(&text).unwrap();
                        if ws_msg.channel == "visualization" {
                            count += 1;
                        }
                    }
                    _ => continue,
                }
            }
        }

        messages_clone.store(count, std::sync::atomic::Ordering::SeqCst);
    });

    data_task.await.unwrap();

    let received = messages_received.load(std::sync::atomic::Ordering::SeqCst);
    let expected_fps = 60.0;
    let expected_messages = (expected_fps * 0.5) as usize; // 500ms test
    let tolerance = expected_messages / 10; // 10% tolerance

    assert!(
        received >= expected_messages - tolerance && received <= expected_messages + tolerance,
        "Expected {}±{} messages, got {}",
        expected_messages, tolerance, received
    );
}

#[tokio::test]
async fn test_binary_protocol_chunked_transfer() {
    init_test_logger();

    // Create large test data (>1MB)
    let large_data = TestDataBuilder::new()
        .with_sine_wave(44100, 1000.0, Duration::from_secs(10))
        .build();

    assert!(large_data.len() > 1024 * 1024, "Test data should be >1MB");

    // Create network message
    let proto_msg = network_message::Payload::AudioData(AudioData {
        data: large_data.clone(),
        format: AudioFormat::Pcm as i32,
        sample_rate: 44100,
        channels: 2,
        bit_depth: 16,
    });

    // Serialize and chunk
    let serialized = BinaryProtocol::serialize_message(&proto_msg).unwrap();
    let chunks = BinaryProtocol::create_chunks(&serialized, 65536); // 64KB chunks

    assert!(chunks.len() > 1, "Large data should be split into multiple chunks");

    // Simulate network transfer with random ordering
    let mut shuffled_chunks = chunks.clone();
    use rand::seq::SliceRandom;
    shuffled_chunks.shuffle(&mut rand::thread_rng());

    // Reconstruct on receiving end
    let mut received_data = std::collections::HashMap::new();
    for chunk in shuffled_chunks {
        received_data.insert(chunk.chunk_id, chunk);
    }

    let mut ordered_chunks: Vec<_> = received_data.into_values().collect();
    ordered_chunks.sort_by_key(|chunk| chunk.sequence_number);

    let reconstructed = BinaryProtocol::reconstruct_chunks(ordered_chunks).unwrap();

    assert_eq!(reconstructed.len(), serialized.len());
    assert_eq!(reconstructed, serialized);

    // Verify message deserialization
    let deserialized = BinaryProtocol::deserialize_message(&reconstructed).unwrap();
    match deserialized.payload {
        Some(network_message::Payload::AudioData(audio_data)) => {
            assert_eq!(audio_data.data.len(), large_data.len());
            assert_eq!(audio_data.data, large_data);
        }
        _ => panic!("Expected AudioData payload"),
    }
}

#[tokio::test]
async fn test_concurrent_clients() {
    init_test_logger();

    let http_port = 8082;
    let ws_port = 8083;

    // Start HTTP and WebSocket servers
    let http_handler = HttpHandler::new();
    let ws_handler = WebSocketHandler::new();

    let http_clone = http_handler.clone();
    let ws_clone = ws_handler.clone();

    // HTTP server
    tokio::spawn(async move {
        http_clone.start(http_port).await;
    });

    // WebSocket server
    tokio::spawn(async move {
        ws_clone.start(ws_port).await;
    });

    // Wait for servers to start
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Create multiple concurrent clients
    let num_clients = 10;
    let client_tasks: Vec<_> = (0..num_clients).map(|client_id| {
        tokio::spawn(async move {
            // HTTP client
            let client = reqwest::Client::new();
            let test_data = TestDataBuilder::new()
                .with_sine_wave(44100, (client_id * 100 + 440) as f32, Duration::from_millis(50))
                .build();

            let metadata = AudioMetadata {
                title: format!("Client {} Test", client_id),
                ..create_test_audio_metadata()
            };

            let metadata_json = serde_json::to_string(&metadata).unwrap();
            let http_response = client
                .post(&format!("http://127.0.0.1:{}/api/audio/upload", http_port))
                .header("X-Audio-Metadata", metadata_json)
                .body(test_data)
                .send()
                .await;

            assert!(http_response.is_ok(), "HTTP upload failed for client {}", client_id);

            // WebSocket client
            let (ws_stream, _) = connect_async(&format!("ws://127.0.0.1:{}", ws_port))
                .await
                .expect("Failed to connect WebSocket");

            let (mut ws_sender, mut ws_receiver) = ws_stream.split();

            // Subscribe
            let subscribe_msg = WebSocketMessage {
                message_type: MessageType::Subscribe as i32,
                channel: format!("client_{}", client_id),
                timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                payload: None,
            };

            let subscribe_json = serde_json::to_string(&subscribe_msg).unwrap();
            ws_sender.send(Message::Text(subscribe_json)).await.unwrap();

            // Wait for response
            let mut received_response = false;
            for _ in 0..10 {
                if let Some(msg) = ws_receiver.next().await {
                    match msg {
                        Ok(Message::Text(_)) => {
                            received_response = true;
                            break;
                        }
                        _ => continue,
                    }
                }
            }

            assert!(received_response, "WebSocket response not received for client {}", client_id);
        })
    }).collect();

    // Wait for all clients to complete
    for task in client_tasks {
        task.await.expect("Client task panicked");
    }
}

#[tokio::test]
async fn test_discovery_service_integration() {
    init_test_logger();

    // Start discovery service
    let discovery = DiscoveryService::new();
    let discovery_clone = discovery.clone();

    tokio::spawn(async move {
        discovery_clone.start().await;
    });

    // Wait for service to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Test service discovery
    let discovered_services = discovery.discover_services(Duration::from_secs(2)).await;

    // Should at least find itself
    assert!(!discovered_services.is_empty());

    // Test service registration
    let test_service = vortex_proto::discovery::ServiceInfo {
        name: "Test Audio Service".to_string(),
        service_type: "audio-backend".to_string(),
        address: "127.0.0.1".to_string(),
        port: 8080,
        metadata: HashMap::from([
            ("version".to_string(), "1.0.0".to_string()),
            ("features".to_string(), "gpu,realtime".to_string()),
        ]),
        last_seen: chrono::Utc::now().timestamp(),
    };

    discovery.register_service(test_service.clone()).await.unwrap();

    // Verify service discovery after registration
    let services = discovery.discover_services(Duration::from_secs(1)).await;
    let found_service = services.iter()
        .find(|s| s.name == test_service.name && s.service_type == test_service.service_type);

    assert!(found_service.is_some(), "Registered service not found");
    let discovered = found_service.unwrap();
    assert_eq!(discovered.address, test_service.address);
    assert_eq!(discovered.port, test_service.port);
}

realtime_test!(test_realtime_message_processing, 10, {
    let handler = WebSocketHandler::new();

    let messages: Vec<_> = (0..100).map(|i| {
        WebSocketMessage {
            message_type: MessageType::Data as i32,
            channel: format!("channel_{}", i % 5),
            timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            payload: Some(websocket_message::Payload::Heartbeat(())),
        }
    }).collect();

    for message in messages {
        let _result = handler.process_message(message).await;
    }
});

gpu_test!(test_gpu_enabled_processing, {
    // This test will only run if GPU tests are enabled
    let large_spectrum = SpectrumData {
        frequency_bins: (0..2048).map(|i| i as f32 * 21.533).collect(),
        magnitudes: vec![0.0f32; 2048],
        sample_rate: 44100,
        fft_size: 4096,
        window_type: "blackman".to_string(),
        timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
    };

    let message = WebSocketMessage {
        message_type: MessageType::Data as i32,
        channel: "gpu_spectrum".to_string(),
        timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
        payload: Some(websocket_message::Payload::SpectrumData(large_spectrum)),
    };

    let handler = WebSocketHandler::new();
    let _result = handler.process_message(message).await;
});