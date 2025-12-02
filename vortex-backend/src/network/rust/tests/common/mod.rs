//! Common testing utilities for vortex network services

use std::collections::HashMap;
use std::sync::Once;
use std::time::Duration;

pub use prost::Message;
pub use vortex_proto::network::*;
pub use vortex_proto::websocket::*;

static INIT: Once = Once::new();

pub fn init_test_logger() {
    INIT.call_once(|| {
        env_logger::builder()
            .filter_level(log::LevelFilter::Debug)
            .is_test(true)
            .init();
    });
}

pub struct TestDataBuilder {
    data: Vec<u8>,
}

impl TestDataBuilder {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
        }
    }

    pub fn with_sine_wave(mut self, sample_rate: u32, frequency: f32, duration: Duration) -> Self {
        let num_samples = (sample_rate as f64 * duration.as_secs_f64()) as usize;
        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin();
            self.data.extend_from_slice(&sample.to_le_bytes());
        }
        self
    }

    pub fn build(self) -> Vec<u8> {
        self.data
    }
}

impl Default for TestDataBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub fn create_test_audio_metadata() -> AudioMetadata {
    AudioMetadata {
        title: "Test Track".to_string(),
        artist: "Test Artist".to_string(),
        album: "Test Album".to_string(),
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
            ("genre".to_string(), "Test".to_string()),
            ("year".to_string(), "2024".to_string()),
        ]),
    }
}

pub fn create_test_spectrum_data() -> SpectrumData {
    SpectrumData {
        frequency_bins: (0..512).map(|i| i as f32 * 43.066).collect(),
        magnitudes: vec![0.0f32; 512],
        sample_rate: 44100,
        fft_size: 1024,
        window_type: "hann".to_string(),
        timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
    }
}

pub fn create_test_websocket_message() -> WebSocketMessage {
    WebSocketMessage {
        message_type: MessageType::Data as i32,
        channel: "visualization".to_string(),
        timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
        payload: Some(websocket_message::Payload::SpectrumData(
            create_test_spectrum_data()
        )),
    }
}

pub async fn setup_test_server() -> tokio::net::TcpListener {
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("Failed to bind test server");

    listener
}

pub fn assert_realtime_constraint(start: std::time::Instant, end: std::time::Instant, max_duration: Duration) {
    let elapsed = end.duration_since(start);
    assert!(
        elapsed <= max_duration,
        "Real-time constraint violated: {} > {:?}",
        elapsed.as_millis(),
        max_duration
    );
}

pub fn assert_audio_quality(original: &[f32], processed: &[f32], max_difference: f32) {
    assert_eq!(
        original.len(),
        processed.len(),
        "Audio buffer length mismatch"
    );

    for (i, (orig, proc)) in original.iter().zip(processed.iter()).enumerate() {
        let diff = (orig - proc).abs();
        assert!(
            diff <= max_difference,
            "Audio quality violation at sample {}: {} > {}",
            i, diff, max_difference
        );
    }
}

#[macro_export]
macro_rules! gpu_test {
    ($test_name:ident, $test_body:block) => {
        #[tokio::test]
        #[cfg(feature = "gpu-tests")]
        async fn $test_name() {
            // Check GPU availability
            if !std::env::var("VORTEX_GPU_TESTS").is_ok() {
                log::info!("Skipping GPU test {} (VORTEX_GPU_TESTS not set)", stringify!($test_name));
                return;
            }

            $test_body
        }
    };
}

#[macro_export]
macro_rules! realtime_test {
    ($test_name:ident, $max_ms:expr, $test_body:block) => {
        #[tokio::test]
        async fn $test_name() {
            let start = std::time::Instant::now();
            $test_body
            let elapsed = start.elapsed();

            assert!(
                elapsed.as_millis() <= $max_ms,
                "Real-time test failed: {}ms > {}ms",
                elapsed.as_millis(),
                $max_ms
            );
        }
    };
}