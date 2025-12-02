use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use vortex_network::protocol::websocket::WebSocketHandler;
use vortex_network::protocol::http::HttpHandler;
use vortex_network::discovery::DiscoveryService;
use vortex_proto::websocket::{WebSocketMessage, MessageType, SpectrumData, websocket_message::Payload};
use std::collections::HashMap;
use std::time::Duration;
use tokio::runtime::Runtime;

fn create_spectrum_data(fft_size: usize) -> SpectrumData {
    SpectrumData {
        frequency_bins: (0..fft_size).map(|i| i as f32 * 44100.0 / fft_size as f32).collect(),
        magnitudes: vec![0.0f32; fft_size],
        sample_rate: 44100,
        fft_size: fft_size as u32,
        window_type: "hann".to_string(),
        timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
    }
}

fn bench_websocket_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("websocket_throughput");

    // Benchmark different message sizes and frequencies
    let rt = Runtime::new().unwrap();
    let handler = rt.block_on(async {
        WebSocketHandler::new()
    });

    for fft_size in [512, 1024, 2048, 4096].iter() {
        let spectrum_data = create_spectrum_data(*fft_size);
        let message_size = serde_json::to_string(&WebSocketMessage {
            message_type: MessageType::Data as i32,
            channel: "spectrum".to_string(),
            timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            payload: Some(Payload::SpectrumData(spectrum_data)),
        }).unwrap().len();

        group.throughput(Throughput::Bytes(message_size as u64));

        group.bench_with_input(
            BenchmarkId::new("spectrum_message_throughput", fft_size),
            fft_size,
            |b, &_fft_size| {
                let spectrum_data = create_spectrum_data(*fft_size);
                let message = WebSocketMessage {
                    message_type: MessageType::Data as i32,
                    channel: "spectrum".to_string(),
                    timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                    payload: Some(Payload::SpectrumData(spectrum_data)),
                };

                b.iter(|| {
                    rt.block_on(async {
                        let handler = handler.clone();
                        handler.process_message(black_box(message.clone())).await
                    });
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_connections(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_connections");

    let rt = Runtime::new().unwrap();

    for num_connections in [10, 50, 100, 200, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_websocket_processing", num_connections),
            num_connections,
            |b, &num_connections| {
                b.to_async(&rt).iter(|| async {
                    let handler = WebSocketHandler::new();
                    let spectrum_data = create_spectrum_data(2048);
                    let message = WebSocketMessage {
                        message_type: MessageType::Data as i32,
                        channel: "spectrum".to_string(),
                        timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                        payload: Some(Payload::SpectrumData(spectrum_data)),
                    };

                    let tasks: Vec<_> = (0..num_connections).map(|_| {
                        let handler = handler.clone();
                        let msg = message.clone();
                        tokio::spawn(async move {
                            handler.process_message(msg).await
                        })
                    }).collect();

                    for task in tasks {
                        task.await.unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_discovery_service(c: &mut Criterion) {
    let mut group = c.benchmark_group("discovery_service");

    let rt = Runtime::new().unwrap();
    let discovery = rt.block_on(async {
        DiscoveryService::new()
    });

    // Benchmark service registration
    group.bench_function("register_service", |b| {
        b.to_async(&rt).iter(|| async {
            let test_service = vortex_proto::discovery::ServiceInfo {
                name: "Benchmark Service".to_string(),
                service_type: "audio-backend".to_string(),
                address: "127.0.0.1".to_string(),
                port: 8080,
                metadata: HashMap::from([
                    ("version".to_string(), "1.0.0".to_string()),
                    ("features".to_string(), "gpu,realtime".to_string()),
                ]),
                last_seen: chrono::Utc::now().timestamp(),
            };

            let discovery = discovery.clone();
            discovery.register_service(black_box(test_service)).await.unwrap();
        });
    });

    // Benchmark service discovery
    group.bench_function("discover_services", |b| {
        b.to_async(&rt).iter(|| async {
            let discovery = discovery.clone();
            let _services = discovery.discover_services(Duration::from_millis(100)).await;
        });
    });

    group.finish();
}

fn bench_message_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("message_serialization");

    // Benchmark JSON serialization for WebSocket messages
    let spectrum_data = create_spectrum_data(2048);
    let ws_message = WebSocketMessage {
        message_type: MessageType::Data as i32,
        channel: "spectrum".to_string(),
        timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
        payload: Some(Payload::SpectrumData(spectrum_data)),
    };

    group.bench_function("serialize_websocket_message", |b| {
        b.iter(|| {
            serde_json::to_string(black_box(&ws_message)).unwrap()
        });
    });

    group.bench_function("deserialize_websocket_message", |b| {
        let json = serde_json::to_string(&ws_message).unwrap();
        b.iter(|| {
            serde_json::from_str::<WebSocketMessage>(black_box(&json)).unwrap()
        });
    });

    // Benchmark Protocol Buffers serialization
    group.bench_function("serialize_proto_message", |b| {
        b.iter(|| {
            black_box(&ws_message).encode_to_vec()
        });
    });

    group.bench_function("deserialize_proto_message", |b| {
        let bytes = ws_message.encode_to_vec();
        b.iter(|| {
            WebSocketMessage::decode(black_box(&*bytes)).unwrap()
        });
    });

    group.finish();
}

fn bench_network_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("network_latency");

    let rt = Runtime::new().unwrap();
    let handler = rt.block_on(async {
        WebSocketHandler::new()
    });

    // Measure end-to-end latency for different message types
    group.bench_function("heartbeat_latency", |b| {
        b.to_async(&rt).iter(|| async {
            let message = WebSocketMessage {
                message_type: MessageType::Heartbeat as i32,
                channel: "control".to_string(),
                timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                payload: Some(Payload::Heartbeat(())),
            };

            let start = std::time::Instant::now();
            let handler = handler.clone();
            handler.process_message(black_box(message)).await;
            black_box(start.elapsed());
        });
    });

    group.bench_function("subscription_latency", |b| {
        b.to_async(&rt).iter(|| async {
            let message = WebSocketMessage {
                message_type: MessageType::Subscribe as i32,
                channel: "visualization".to_string(),
                timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                payload: Some(Payload::SubscribeRequest(
                    vortex_proto::websocket::websocket_message::SubscribeRequest {
                        channel: "visualization".to_string(),
                        subscription_type: "spectrum".to_string(),
                        frequency: 60.0,
                        parameters: HashMap::new(),
                    }
                )),
            };

            let start = std::time::Instant::now();
            let handler = handler.clone();
            handler.process_message(black_box(message)).await;
            black_box(start.elapsed());
        });
    });

    group.finish();
}

fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");

    group.bench_function("allocate_large_spectrum", |b| {
        b.iter(|| {
            black_box(create_spectrum_data(16384))
        });
    });

    group.bench_function("allocate_many_small_messages", |b| {
        b.iter(|| {
            let messages: Vec<_> = (0..1000).map(|i| {
                WebSocketMessage {
                    message_type: MessageType::Data as i32,
                    channel: format!("channel_{}", i),
                    timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                    payload: Some(Payload::Heartbeat(())),
                }
            }).collect();
            black_box(messages)
        });
    });

    group.finish();
}

fn bench_subscriptions(c: &mut Criterion) {
    let mut group = c.benchmark_group("subscriptions");

    let rt = Runtime::new().unwrap();

    for num_subscriptions in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("manage_subscriptions", num_subscriptions),
            num_subscriptions,
            |b, &num_subscriptions| {
                b.to_async(&rt).iter(|| async {
                    let handler = WebSocketHandler::new();

                    // Create subscriptions
                    let subscribe_tasks: Vec<_> = (0..num_subscriptions).map(|i| {
                        let handler = handler.clone();
                        tokio::spawn(async move {
                            let message = WebSocketMessage {
                                message_type: MessageType::Subscribe as i32,
                                channel: format!("channel_{}", i),
                                timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                                payload: Some(Payload::SubscribeRequest(
                                    vortex_proto::websocket::websocket_message::SubscribeRequest {
                                        channel: format!("channel_{}", i),
                                        subscription_type: "spectrum".to_string(),
                                        frequency: 60.0,
                                        parameters: HashMap::new(),
                                    }
                                )),
                            };
                            handler.process_message(message).await
                        })
                    }).collect();

                    for task in subscribe_tasks {
                        task.await.unwrap();
                    }

                    // Send broadcast message
                    let spectrum_data = create_spectrum_data(2048);
                    let broadcast_message = WebSocketMessage {
                        message_type: MessageType::Data as i32,
                        channel: "broadcast".to_string(),
                        timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                        payload: Some(Payload::SpectrumData(spectrum_data)),
                    };

                    let _result = handler.process_message(broadcast_message).await;
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_websocket_throughput,
    bench_concurrent_connections,
    bench_discovery_service,
    bench_message_serialization,
    bench_network_latency,
    bench_memory_allocation,
    bench_subscriptions
);
criterion_main!(benches);