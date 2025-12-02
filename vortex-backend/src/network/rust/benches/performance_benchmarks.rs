use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use vortex_network::protocol::binary::BinaryProtocol;
use vortex_network::protocol::websocket::WebSocketHandler;
use vortex_proto::network::{AudioMetadata, AudioFormat, ProcessingChain, ProcessingStep, ProcessingType, AudioData};
use std::collections::HashMap;
use std::time::Duration;

fn create_test_metadata() -> AudioMetadata {
    AudioMetadata {
        title: "Performance Test Track".to_string(),
        artist: "Performance Test Artist".to_string(),
        album: "Performance Test Album".to_string(),
        duration: Some(prost_types::Duration {
            seconds: 180,
            nanos: 0,
        }),
        format: AudioFormat::Pcm as i32,
        sample_rate: 44100,
        bit_depth: 16,
        channels: 2,
        bitrate: 320000,
        metadata: HashMap::from([
            ("genre".to_string(), "Performance Test".to_string()),
            ("year".to_string(), "2024".to_string()),
        ]),
    }
}

fn create_test_audio_data(size_bytes: usize) -> Vec<u8> {
    vec![0u8; size_bytes]
}

fn create_test_processing_chain(num_steps: usize) -> ProcessingChain {
    let steps: Vec<_> = (0..num_steps).map(|i| {
        ProcessingStep {
            id: format!("step_{}", i),
            step_type: ProcessingType::Equalizer as i32,
            parameters: HashMap::from([
                ("frequencies".to_string(), "[100,1000,10000]".to_string()),
                ("gains".to_string(), "[0.0,2.0,-1.0]".to_string()),
                ("q".to_string(), "[1.0,1.414,1.0]".to_string()),
            ]),
            enabled: true,
            order: i as u32,
        }
    }).collect();

    ProcessingChain {
        id: "perf_test_chain".to_string(),
        name: "Performance Test Chain".to_string(),
        steps,
        enabled: true,
    }
}

fn bench_protocol_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("protocol_serialization");

    // Benchmark different message sizes
    for size in [1024, 4096, 16384, 65536, 262144].iter() {
        group.bench_with_input(
            BenchmarkId::new("serialize_audio_data", size),
            size,
            |b, &size| {
                let audio_data = AudioData {
                    data: create_test_audio_data(size),
                    format: AudioFormat::Pcm as i32,
                    sample_rate: 44100,
                    channels: 2,
                    bit_depth: 16,
                };

                let proto_msg = vortex_proto::network::network_message::Payload::AudioData(audio_data);

                b.iter(|| {
                    BinaryProtocol::serialize_message(black_box(&proto_msg)).unwrap()
                });
            },
        );
    }

    // Benchmark metadata serialization
    group.bench_function("serialize_metadata", |b| {
        let metadata = create_test_metadata();
        let proto_msg = vortex_proto::network::network_message::Payload::AudioMetadata(metadata);

        b.iter(|| {
            BinaryProtocol::serialize_message(black_box(&proto_msg)).unwrap()
        });
    });

    // Benchmark processing chain serialization
    for num_steps in [1, 5, 10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("serialize_processing_chain", num_steps),
            num_steps,
            |b, &num_steps| {
                let chain = create_test_processing_chain(num_steps);
                let proto_msg = vortex_proto::network::network_message::Payload::ProcessingChain(chain);

                b.iter(|| {
                    BinaryProtocol::serialize_message(black_box(&proto_msg)).unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_protocol_deserialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("protocol_deserialization");

    // Pre-serialize data for deserialization benchmarks
    let audio_data = AudioData {
        data: create_test_audio_data(16384),
        format: AudioFormat::Pcm as i32,
        sample_rate: 44100,
        channels: 2,
        bit_depth: 16,
    };
    let serialized_audio = BinaryProtocol::serialize_message(
        &vortex_proto::network::network_message::Payload::AudioData(audio_data)
    ).unwrap();

    group.bench_function("deserialize_audio_data", |b| {
        b.iter(|| {
            BinaryProtocol::deserialize_message(black_box(&serialized_audio)).unwrap()
        });
    });

    let metadata = create_test_metadata();
    let serialized_metadata = BinaryProtocol::serialize_message(
        &vortex_proto::network::network_message::Payload::AudioMetadata(metadata)
    ).unwrap();

    group.bench_function("deserialize_metadata", |b| {
        b.iter(|| {
            BinaryProtocol::deserialize_message(black_box(&serialized_metadata)).unwrap()
        });
    });

    group.finish();
}

fn bench_chunking(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunking");

    for size in [65536, 262144, 1048576, 4194304].iter() {
        group.bench_with_input(
            BenchmarkId::new("create_chunks", size),
            size,
            |b, &size| {
                let data = create_test_audio_data(size);

                b.iter(|| {
                    BinaryProtocol::create_chunks(black_box(&data), 65536)
                });
            },
        );
    }

    // Benchmark chunk reconstruction
    let data = create_test_audio_data(1048576);
    let chunks = BinaryProtocol::create_chunks(&data, 65536);

    group.bench_function("reconstruct_chunks", |b| {
        b.iter(|| {
            BinaryProtocol::reconstruct_chunks(black_box(chunks.clone())).unwrap()
        });
    });

    group.finish();
}

fn bench_websocket_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("websocket_processing");

    let handler = WebSocketHandler::new();

    // Create test spectrum data
    let spectrum_data = vortex_proto::websocket::SpectrumData {
        frequency_bins: (0..2048).map(|i| i as f32 * 21.533).collect(),
        magnitudes: vec![0.0f32; 2048],
        sample_rate: 44100,
        fft_size: 2048,
        window_type: "hann".to_string(),
        timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
    };

    let ws_message = vortex_proto::websocket::WebSocketMessage {
        message_type: vortex_proto::websocket::MessageType::Data as i32,
        channel: "spectrum".to_string(),
        timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
        payload: Some(vortex_proto::websocket::websocket_message::Payload::SpectrumData(spectrum_data)),
    };

    group.bench_function("process_spectrum_message", |b| {
        b.iter(|| {
            let handler = handler.clone();
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                handler.process_message(black_box(ws_message.clone())).await
            })
        });
    });

    // Benchmark concurrent message processing
    group.bench_function("concurrent_message_processing", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let handler = WebSocketHandler::new();
                let tasks: Vec<_> = (0..10).map(|_| {
                    let handler = handler.clone();
                    let msg = ws_message.clone();
                    tokio::spawn(async move {
                        handler.process_message(msg).await
                    })
                }).collect();

                for task in tasks {
                    task.await.unwrap();
                }
            });
        });
    });

    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    group.bench_function("large_message_allocation", |b| {
        b.iter(|| {
            let large_data = create_test_audio_data(10_000_000); // 10MB
            let audio_data = AudioData {
                data: large_data,
                format: AudioFormat::Pcm as i32,
                sample_rate: 44100,
                channels: 2,
                bit_depth: 16,
            };

            let proto_msg = vortex_proto::network::network_message::Payload::AudioData(audio_data);
            let _serialized = BinaryProtocol::serialize_message(&proto_msg).unwrap();
        });
    });

    group.bench_function("metadata_with_large_strings", |b| {
        b.iter(|| {
            let large_metadata = AudioMetadata {
                title: "A".repeat(10000),
                artist: "B".repeat(10000),
                album: "C".repeat(10000),
                ..create_test_metadata()
            };

            let proto_msg = vortex_proto::network::network_message::Payload::AudioMetadata(large_metadata);
            let _serialized = BinaryProtocol::serialize_message(&proto_msg).unwrap();
        });
    });

    group.finish();
}

fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");

    // Test compression efficiency
    let audio_data = create_test_audio_data(1048576);
    let audio_msg = AudioData {
        data: audio_data,
        format: AudioFormat::Pcm as i32,
        sample_rate: 44100,
        channels: 2,
        bit_depth: 16,
    };

    let proto_msg = vortex_proto::network::network_message::Payload::AudioData(audio_msg);

    group.bench_function("compress_large_audio_data", |b| {
        b.iter(|| {
            BinaryProtocol::serialize_message(black_box(&proto_msg)).unwrap()
        });
    });

    // Test with repeating patterns (better compression)
    let repeating_data = (0..255).cycle().take(1048576).collect::<Vec<u8>>();
    let repeating_audio = AudioData {
        data: repeating_data,
        format: AudioFormat::Pcm as i32,
        sample_rate: 44100,
        channels: 2,
        bit_depth: 16,
    };

    let repeating_msg = vortex_proto::network::network_message::Payload::AudioData(repeating_audio);

    group.bench_function("compress_repeating_pattern", |b| {
        b.iter(|| {
            BinaryProtocol::serialize_message(black_box(&repeating_msg)).unwrap()
        });
    });

    group.finish();
}

fn bench_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation");

    let valid_metadata = create_test_metadata();

    group.bench_function("validate_metadata", |b| {
        b.iter(|| {
            vortex_network::protocol::http::HttpHandler::validate_audio_metadata(black_box(&valid_metadata)).unwrap()
        });
    });

    // Benchmark processing chain validation
    for num_steps in [1, 5, 10, 25, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("validate_processing_chain", num_steps),
            num_steps,
            |b, &num_steps| {
                let chain = create_test_processing_chain(num_steps);

                b.iter(|| {
                    BinaryProtocol::validate_processing_chain(black_box(&chain)).unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_protocol_serialization,
    bench_protocol_deserialization,
    bench_chunking,
    bench_websocket_processing,
    bench_memory_usage,
    bench_compression,
    bench_validation
);
criterion_main!(benches);