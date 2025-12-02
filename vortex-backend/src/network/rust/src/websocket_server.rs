use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, tungstenite::protocol::Message, WebSocketStream};
use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;
use crate::protocol::websocket::{WebSocketMessage, MessageType, Payload, SpectrumData};
use crate::protocol::binary::BinaryProtocol;

/// Maximum number of concurrent WebSocket connections
const MAX_CONNECTIONS: usize = 1000;

/// WebSocket message rate limiting
const MAX_MESSAGES_PER_SECOND: u32 = 60;

/// Connection information
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    pub id: String,
    pub client_info: ClientInfo,
    pub subscriptions: HashMap<String, SubscriptionInfo>,
    pub last_activity: SystemTime,
    pub message_count: u32,
    pub created_at: SystemTime,
}

/// Client information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    pub user_agent: Option<String>,
    pub ip_address: String,
    pub client_type: String, // "web", "mobile", "desktop"
    pub version: Option<String>,
}

/// Subscription information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionInfo {
    pub channel: String,
    pub subscription_type: String, // "spectrum", "levels", "position", etc.
    pub frequency: f64, // Updates per second
    pub parameters: HashMap<String, serde_json::Value>,
    pub last_sent: SystemTime,
}

/// WebSocket server for real-time audio data streaming
#[derive(Debug)]
pub struct WebSocketServer {
    connections: Arc<RwLock<HashMap<String, WebSocketStream<TcpStream>>>>,
    connection_infos: Arc<RwLock<HashMap<String, ConnectionInfo>>>,
    binary_protocol: Arc<BinaryProtocol>,
    port: u16,
    running: Arc<RwLock<bool>>,
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub port: u16,
    pub max_connections: usize,
    pub heartbeat_interval: Duration,
    pub message_buffer_size: usize,
    pub enable_compression: bool,
    pub enable_metrics: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 8081,
            max_connections: MAX_CONNECTIONS,
            heartbeat_interval: Duration::from_secs(30),
            message_buffer_size: 1024 * 1024, // 1MB
            enable_compression: true,
            enable_metrics: true,
        }
    }
}

impl WebSocketServer {
    pub fn new(config: ServerConfig) -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            connection_infos: Arc::new(RwLock::new(HashMap::new())),
            binary_protocol: Arc::new(BinaryProtocol::new()),
            port: config.port,
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the WebSocket server
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        {
            let mut running = self.running.write().await;
            *running = true;
        }

        let listener = TcpListener::bind(format!("0.0.0.0:{}", self.port)).await?;
        println!("WebSocket server listening on port {}", self.port);

        while *self.running.read().await {
            match listener.accept().await {
                Ok((stream, addr)) => {
                    let ip = addr.ip().to_string();
                    println!("New connection from: {}", ip);

                    // Check connection limit
                    {
                        let connections = self.connections.read().await;
                        if connections.len() >= MAX_CONNECTIONS {
                            println!("Connection limit reached, rejecting from: {}", ip);
                            continue;
                        }
                    }

                    // Handle new connection
                    let server = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = server.handle_connection(stream, ip).await {
                            println!("Connection error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    if *self.running.read().await {
                        println!("Error accepting connection: {}", e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Stop the WebSocket server
    pub async fn stop(&self) {
        {
            let mut running = self.running.write().await;
            *running = false;
        }

        // Close all connections
        let connections = self.connections.read().await;
        for (id, _stream) in connections.iter() {
            println!("Closing connection: {}", id);
            // Note: In a real implementation, we'd need to close the actual streams
        }
    }

    /// Handle individual WebSocket connection
    async fn handle_connection(&self, stream: TcpStream, client_ip: String) -> Result<(), Box<dyn std::error::Error>> {
        let ws_stream = accept_async(stream).await?;
        let (mut sender, mut receiver) = ws_stream.split();

        // Generate connection ID
        let connection_id = Uuid::new_v4().to_string();

        // Create connection info
        let client_info = ClientInfo {
            user_agent: None, // Would be extracted from HTTP headers
            ip_address: client_ip,
            client_type: "unknown".to_string(),
            version: None,
        };

        let connection_info = ConnectionInfo {
            id: connection_id.clone(),
            client_info,
            subscriptions: HashMap::new(),
            last_activity: SystemTime::now(),
            message_count: 0,
            created_at: SystemTime::now(),
        };

        // Store connection
        {
            let mut connections = self.connections.write().await;
            let mut connection_infos = self.connection_infos.write().await;

            // Clone the stream for storage (this is simplified - would need proper stream handling)
            // connections.insert(connection_id.clone(), ws_stream);
            connection_infos.insert(connection_id.clone(), connection_info);
        }

        println!("WebSocket connection established: {}", connection_id);

        // Handle messages from client
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Err(e) = self.handle_text_message(&connection_id, &text, &mut sender).await {
                        println!("Error handling text message: {}", e);
                        break;
                    }
                }
                Ok(Message::Binary(data)) => {
                    if let Err(e) = self.handle_binary_message(&connection_id, &data, &mut sender).await {
                        println!("Error handling binary message: {}", e);
                        break;
                    }
                }
                Ok(Message::Close(_)) => {
                    println!("Connection closed by client: {}", connection_id);
                    break;
                }
                Err(e) => {
                    println!("WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }

            // Update last activity
            {
                let mut connection_infos = self.connection_infos.write().await;
                if let Some(info) = connection_infos.get_mut(&connection_id) {
                    info.last_activity = SystemTime::now();
                    info.message_count += 1;
                }
            }
        }

        // Cleanup connection
        {
            let mut connections = self.connections.write().await;
            let mut connection_infos = self.connection_infos.write().await;
            connections.remove(&connection_id);
            connection_infos.remove(&connection_id);
        }

        println!("WebSocket connection closed: {}", connection_id);
        Ok(())
    }

    /// Handle text message from client
    async fn handle_text_message(
        &self,
        connection_id: &str,
        text: &str,
        sender: &mut futures_util::stream::SplitSink<WebSocketStream<TcpStream>, Message>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Parse JSON message
        let ws_message: WebSocketMessage = serde_json::from_str(text)?;

        match ws_message.message_type() {
            MessageType::Subscribe => {
                self.handle_subscription(connection_id, ws_message).await?;
            }
            MessageType::Unsubscribe => {
                self.handle_unsubscription(connection_id, ws_message).await?;
            }
            MessageType::Heartbeat => {
                self.handle_heartbeat(connection_id, sender).await?;
            }
            MessageType::Control => {
                self.handle_control_message(connection_id, ws_message).await?;
            }
            _ => {
                println!("Unhandled message type: {:?}", ws_message.message_type());
            }
        }

        Ok(())
    }

    /// Handle binary message from client
    async fn handle_binary_message(
        &self,
        connection_id: &str,
        data: &[u8],
        sender: &mut futures_util::stream::SplitSink<WebSocketStream<TcpStream>, Message>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Try to decode binary message using our protocol
        if let Some(network_message) = self.binary_protocol.deserialize_message(data) {
            self.handle_network_message(connection_id, network_message, sender).await?;
        } else {
            println!("Invalid binary message received from: {}", connection_id);
        }

        Ok(())
    }

    /// Handle subscription request
    async fn handle_subscription(&self, connection_id: &str, message: WebSocketMessage) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(Payload::SubscribeRequest(req)) = message.payload() {
            let subscription_info = SubscriptionInfo {
                channel: req.channel.clone(),
                subscription_type: req.subscription_type.clone(),
                frequency: req.frequency,
                parameters: req.parameters,
                last_sent: SystemTime::now(),
            };

            // Store subscription
            {
                let mut connection_infos = self.connection_infos.write().await;
                if let Some(info) = connection_infos.get_mut(connection_id) {
                    info.subscriptions.insert(req.channel.clone(), subscription_info);
                }
            }

            // Send acknowledgment
            let ack = WebSocketMessage {
                message_type: MessageType::SubscriptionAck as i32,
                channel: req.channel.clone(),
                timestamp: current_timestamp(),
                payload: Some(Payload::SubscriptionAck(())),
            };

            // Send ack back to client (would need the actual sender)
            println!("Subscription added: {} -> {}", connection_id, req.channel);
        }

        Ok(())
    }

    /// Handle unsubscription request
    async fn handle_unsubscription(&self, connection_id: &str, message: WebSocketMessage) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(Payload::UnsubscribeRequest(req)) = message.payload() {
            // Remove subscription
            {
                let mut connection_infos = self.connection_infos.write().await;
                if let Some(info) = connection_infos.get_mut(connection_id) {
                    info.subscriptions.remove(&req.channel);
                }
            }

            println!("Subscription removed: {} -> {}", connection_id, req.channel);
        }

        Ok(())
    }

    /// Handle heartbeat
    async fn handle_heartbeat(
        &self,
        connection_id: &str,
        sender: &mut futures_util::stream::SplitSink<WebSocketStream<TcpStream>, Message>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let response = WebSocketMessage {
            message_type: MessageType::Heartbeat as i32,
            channel: "control".to_string(),
            timestamp: current_timestamp(),
            payload: Some(Payload::Heartbeat(())),
        };

        let json = serde_json::to_string(&response)?;
        sender.send(Message::Text(json)).await?;

        Ok(())
    }

    /// Handle control message
    async fn handle_control_message(&self, connection_id: &str, message: WebSocketMessage) -> Result<(), Box<dyn std::error::Error>> {
        println!("Control message received from {}: {:?}", connection_id, message);
        // Handle various control commands
        Ok(())
    }

    /// Handle network message (binary protocol)
    async fn handle_network_message(
        &self,
        connection_id: &str,
        message: network_message::NetworkMessage,
        sender: &mut futures_util::stream::SplitSink<WebSocketStream<TcpStream>, Message>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("Network message received from {}: {:?}", connection_id, message.message_type());
        // Handle network protocol messages
        Ok(())
    }

    /// Broadcast data to all subscribed clients
    pub async fn broadcast(&self, channel: &str, data: &SpectrumData) -> Result<(), Box<dyn std::error::Error>> {
        let message = WebSocketMessage {
            message_type: MessageType::Data as i32,
            channel: channel.to_string(),
            timestamp: current_timestamp(),
            payload: Some(Payload::SpectrumData(data.clone())),
        };

        let json = serde_json::to_string(&message)?;

        // Find all connections subscribed to this channel
        let connection_infos = self.connection_infos.read().await;
        let mut subscribed_connections = Vec::new();

        for (id, info) in connection_infos.iter() {
            if info.subscriptions.contains_key(channel) {
                subscribed_connections.push(id.clone());
            }
        }

        // Send to subscribed connections
        let connections = self.connections.read().await;
        for connection_id in subscribed_connections {
            if let Some(_stream) = connections.get(&connection_id) {
                // Send message (simplified - would need proper stream handling)
                println!("Broadcasting to {} on channel {}", connection_id, channel);
            }
        }

        Ok(())
    }

    /// Get server statistics
    pub async fn get_stats(&self) -> ServerStats {
        let connection_infos = self.connection_infos.read().await;
        let current_time = SystemTime::now();

        let mut total_subscriptions = 0;
        let mut active_connections = 0;
        let mut oldest_connection: Option<SystemTime> = None;
        let mut newest_connection: Option<SystemTime> = None;

        for info in connection_infos.values() {
            // Check if connection is active (no activity for more than 5 minutes)
            if let Ok(duration) = current_time.duration_since(info.last_activity) {
                if duration < Duration::from_secs(300) {
                    active_connections += 1;
                }
            }

            total_subscriptions += info.subscriptions.len();

            if oldest_connection.is_none() || info.created_at < oldest_connection.unwrap() {
                oldest_connection = Some(info.created_at);
            }
            if newest_connection.is_none() || info.created_at > newest_connection.unwrap() {
                newest_connection = Some(info.created_at);
            }
        }

        ServerStats {
            total_connections: connection_infos.len(),
            active_connections,
            total_subscriptions,
            oldest_connection,
            newest_connection,
            uptime: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

/// Server statistics
#[derive(Debug, Serialize)]
pub struct ServerStats {
    pub total_connections: usize,
    pub active_connections: usize,
    pub total_subscriptions: usize,
    pub oldest_connection: Option<SystemTime>,
    pub newest_connection: Option<SystemTime>,
    pub uptime: u64,
}

/// Helper function to get current timestamp in microseconds
fn current_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as i64
}

/// Clone implementation for WebSocketServer
impl Clone for WebSocketServer {
    fn clone(&self) -> Self {
        Self {
            connections: Arc::clone(&self.connections),
            connection_infos: Arc::clone(&self.connection_infos),
            binary_protocol: Arc::clone(&self.binary_protocol),
            port: self.port,
            running: Arc::clone(&self.running),
        }
    }
}