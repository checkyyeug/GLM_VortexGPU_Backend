use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageType {
    Unknown = 0,
    Subscribe = 1,
    Unsubscribe = 2,
    Data = 3,
    Control = 4,
    Heartbeat = 5,
    SubscriptionAck = 6,
    Error = 7,
}

impl MessageType {
    pub fn from_i32(value: i32) -> Self {
        match value {
            1 => MessageType::Subscribe,
            2 => MessageType::Unsubscribe,
            3 => MessageType::Data,
            4 => MessageType::Control,
            5 => MessageType::Heartbeat,
            6 => MessageType::SubscriptionAck,
            7 => MessageType::Error,
            _ => MessageType::Unknown,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketMessage {
    #[serde(rename = "type")]
    pub message_type: i32,
    pub channel: String,
    pub timestamp: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<Payload>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "payload_type", content = "data")]
pub enum Payload {
    #[serde(rename = "subscribe_request")]
    SubscribeRequest(SubscribeRequest),
    #[serde(rename = "unsubscribe_request")]
    UnsubscribeRequest(UnsubscribeRequest),
    #[serde(rename = "spectrum_data")]
    SpectrumData(SpectrumData),
    #[serde(rename = "audio_levels")]
    AudioLevels(AudioLevels),
    #[serde(rename = "position_info")]
    PositionInfo(PositionInfo),
    #[serde(rename = "control_command")]
    ControlCommand(ControlCommand),
    #[serde(rename = "heartbeat")]
    Heartbeat(()),
    #[serde(rename = "subscription_ack")]
    SubscriptionAck(()),
    #[serde(rename = "error")]
    Error(ErrorInfo),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscribeRequest {
    pub channel: String,
    pub subscription_type: String,
    pub frequency: f64,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsubscribeRequest {
    pub channel: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumData {
    pub frequency_bins: Vec<f32>,
    pub magnitudes: Vec<f32>,
    pub sample_rate: f64,
    pub fft_size: u32,
    pub window_type: String,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioLevels {
    pub left: f32,
    pub right: f32,
    pub peak_left: f32,
    pub peak_right: f32,
    pub rms_left: f32,
    pub rms_right: f32,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionInfo {
    pub current_position: f64,
    pub total_duration: f64,
    pub is_playing: bool,
    pub playback_rate: f64,
    pub loop_enabled: bool,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlCommand {
    pub command_type: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    pub error_code: i32,
    pub error_message: String,
    pub details: Option<HashMap<String, serde_json::Value>>,
    pub timestamp: i64,
}

impl WebSocketMessage {
    pub fn new(message_type: MessageType, channel: String) -> Self {
        Self {
            message_type: message_type as i32,
            channel,
            timestamp: current_timestamp(),
            payload: None,
        }
    }

    pub fn with_payload(message_type: MessageType, channel: String, payload: Payload) -> Self {
        Self {
            message_type: message_type as i32,
            channel,
            timestamp: current_timestamp(),
            payload: Some(payload),
        }
    }

    pub fn message_type(&self) -> MessageType {
        MessageType::from_i32(self.message_type)
    }
}

fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as i64
}

/// WebSocket message builder for easier message creation
pub struct MessageBuilder {
    message_type: MessageType,
    channel: String,
}

impl MessageBuilder {
    pub fn new(message_type: MessageType, channel: impl Into<String>) -> Self {
        Self {
            message_type,
            channel: channel.into(),
        }
    }

    pub fn heartbeat() -> Self {
        Self::new(MessageType::Heartbeat, "control")
    }

    pub fn data(channel: impl Into<String>) -> Self {
        Self::new(MessageType::Data, channel)
    }

    pub fn subscription_ack(channel: impl Into<String>) -> Self {
        Self::new(MessageType::SubscriptionAck, channel)
    }

    pub fn error(channel: impl Into<String>) -> Self {
        Self::new(MessageType::Error, channel)
    }

    pub fn with_spectrum_data(self, data: SpectrumData) -> WebSocketMessage {
        WebSocketMessage::with_payload(self.message_type, self.channel, Payload::SpectrumData(data))
    }

    pub fn with_audio_levels(self, levels: AudioLevels) -> WebSocketMessage {
        WebSocketMessage::with_payload(self.message_type, self.channel, Payload::AudioLevels(levels))
    }

    pub fn with_position_info(self, info: PositionInfo) -> WebSocketMessage {
        WebSocketMessage::with_payload(self.message_type, self.channel, Payload::PositionInfo(info))
    }

    pub fn with_control_command(self, command: ControlCommand) -> WebSocketMessage {
        WebSocketMessage::with_payload(self.message_type, self.channel, Payload::ControlCommand(command))
    }

    pub fn with_heartbeat(self) -> WebSocketMessage {
        WebSocketMessage::with_payload(self.message_type, self.channel, Payload::Heartbeat(()))
    }

    pub fn with_subscription_ack(self) -> WebSocketMessage {
        WebSocketMessage::with_payload(self.message_type, self.channel, Payload::SubscriptionAck(()))
    }

    pub fn with_error(self, error: ErrorInfo) -> WebSocketMessage {
        WebSocketMessage::with_payload(self.message_type, self.channel, Payload::Error(error))
    }
}

/// WebSocket message validator
pub struct MessageValidator;

impl MessageValidator {
    pub fn validate_message(message: &WebSocketMessage) -> Result<(), ValidationError> {
        // Validate timestamp
        let now = current_timestamp();
        if (message.timestamp - now).abs() > 300_000_000 { // 5 minutes
            return Err(ValidationError::InvalidTimestamp);
        }

        // Validate channel name
        if message.channel.is_empty() || message.channel.len() > 256 {
            return Err(ValidationError::InvalidChannel);
        }

        // Validate payload based on message type
        if let Some(ref payload) = message.payload {
            Self::validate_payload(message.message_type(), payload)?;
        }

        Ok(())
    }

    fn validate_payload(message_type: MessageType, payload: &Payload) -> Result<(), ValidationError> {
        match (message_type, payload) {
            (MessageType::Subscribe, Payload::SubscribeRequest(req)) => {
                if req.channel.is_empty() || req.subscription_type.is_empty() {
                    return Err(ValidationError::InvalidSubscription);
                }
                if req.frequency <= 0.0 || req.frequency > 1000.0 {
                    return Err(ValidationError::InvalidFrequency);
                }
            }
            (MessageType::Data, Payload::SpectrumData(data)) => {
                if data.frequency_bins.len() != data.magnitudes.len() {
                    return Err(ValidationError::InvalidSpectrumData);
                }
                if data.sample_rate <= 0.0 {
                    return Err(ValidationError::InvalidSampleRate);
                }
            }
            (MessageType::Data, Payload::AudioLevels(levels)) => {
                if levels.left < -1.0 || levels.left > 1.0 ||
                   levels.right < -1.0 || levels.right > 1.0 {
                    return Err(ValidationError::InvalidAudioLevels);
                }
            }
            _ => {} // Other combinations are valid
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum ValidationError {
    InvalidTimestamp,
    InvalidChannel,
    InvalidSubscription,
    InvalidFrequency,
    InvalidSpectrumData,
    InvalidSampleRate,
    InvalidAudioLevels,
    InvalidPayload,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::InvalidTimestamp => write!(f, "Invalid timestamp"),
            ValidationError::InvalidChannel => write!(f, "Invalid channel name"),
            ValidationError::InvalidSubscription => write!(f, "Invalid subscription request"),
            ValidationError::InvalidFrequency => write!(f, "Invalid frequency"),
            ValidationError::InvalidSpectrumData => write!(f, "Invalid spectrum data"),
            ValidationError::InvalidSampleRate => write!(f, "Invalid sample rate"),
            ValidationError::InvalidAudioLevels => write!(f, "Invalid audio levels"),
            ValidationError::InvalidPayload => write!(f, "Invalid payload"),
        }
    }
}

impl std::error::Error for ValidationError {}