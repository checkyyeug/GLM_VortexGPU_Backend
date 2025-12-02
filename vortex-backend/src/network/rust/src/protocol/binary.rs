use prost::Message;
use std::collections::HashMap;
use std::io::{Read, Write};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use super::network_message;

/// Binary protocol for efficient network communication
pub struct BinaryProtocol {
    compression_enabled: bool,
    compression_level: u32,
    checksum_enabled: bool,
}

impl BinaryProtocol {
    pub fn new() -> Self {
        Self {
            compression_enabled: true,
            compression_level: 6,
            checksum_enabled: true,
        }
    }

    /// Serialize a network message to binary format
    pub fn serialize_message(&message: &network_message::NetworkMessage) -> Result<Vec<u8>, ProtocolError> {
        // Serialize protocol buffer
        let protobuf_data = message.encode_to_vec();

        // Apply compression if enabled
        let payload_data = if self.compression_enabled {
            self.compress_data(&protobuf_data)?
        } else {
            protobuf_data
        };

        // Create binary message header
        let header = MessageHeader {
            version: 1,
            message_type: message.r#type() as u16,
            flags: if self.compression_enabled { 0x01 } else { 0x00 },
            payload_size: payload_data.len() as u32,
            timestamp: self.current_timestamp(),
            checksum: 0, // Will be calculated
        };

        // Calculate checksum
        let mut header = header;
        if self.checksum_enabled {
            header.checksum = self.calculate_checksum(&header, &payload_data);
        }

        // Serialize complete message
        let mut buffer = Vec::new();
        self.write_header(&mut buffer, &header)?;
        buffer.extend_from_slice(&payload_data);

        Ok(buffer)
    }

    /// Deserialize binary data to network message
    pub fn deserialize_message(&data: &[u8]) -> Result<network_message::NetworkMessage, ProtocolError> {
        if data.len() < std::mem::size_of::<MessageHeader>() {
            return Err(ProtocolError::InsufficientData);
        }

        // Read header
        let (header, payload_offset) = self.read_header(data)?;

        // Validate header
        self.validate_header(&header)?;

        // Verify checksum if present
        if (header.flags & 0x02) != 0 && self.checksum_enabled {
            let payload_data = &data[payload_offset..];
            let calculated_checksum = self.calculate_checksum(&header, payload_data);
            if calculated_checksum != header.checksum {
                return Err(ProtocolError::ChecksumMismatch);
            }
        }

        // Extract payload
        let payload_end = payload_offset + header.payload_size as usize;
        if data.len() < payload_end {
            return Err(ProtocolError::InsufficientData);
        }
        let payload_data = &data[payload_offset..payload_end];

        // Decompress if needed
        let protobuf_data = if (header.flags & 0x01) != 0 {
            self.decompress_data(payload_data)?
        } else {
            payload_data.to_vec()
        };

        // Parse protocol buffer
        network_message::NetworkMessage::decode(&protobuf_data[..])
            .map_err(ProtocolError::ProtobufDecode)
    }

    /// Create chunks from large data
    pub fn create_chunks(&data: &[u8], chunk_size: usize) -> Vec<NetworkChunk> {
        let total_chunks = (data.len() + chunk_size - 1) / chunk_size;
        let mut chunks = Vec::with_capacity(total_chunks);

        for (chunk_index, start) in (0..data.len()).step_by(chunk_size).enumerate() {
            let end = std::cmp::min(start + chunk_size, data.len());
            let chunk_data = data[start..end].to_vec();

            let chunk = NetworkChunk {
                chunk_id: 1,
                total_chunks: total_chunks as u32,
                sequence_number: chunk_index as u32,
                data: chunk_data,
                checksum: if self.checksum_enabled {
                    self.calculate_crc32(&data[start..end])
                } else {
                    0
                },
            };

            chunks.push(chunk);
        }

        chunks
    }

    /// Reconstruct data from chunks
    pub fn reconstruct_chunks(&chunks: &[NetworkChunk]) -> Result<Vec<u8>, ProtocolError> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        // Validate chunk consistency
        let first_chunk = &chunks[0];
        let total_chunks = first_chunk.total_chunks as usize;

        if chunks.len() != total_chunks {
            return Err(ProtocolError::IncompleteChunks);
        }

        // Verify checksums
        if self.checksum_enabled {
            for chunk in chunks {
                let calculated_checksum = self.calculate_crc32(&chunk.data);
                if calculated_checksum != chunk.checksum {
                    return Err(ProtocolError::ChecksumMismatch);
                }
            }
        }

        // Sort chunks by sequence number
        let mut sorted_chunks = chunks.to_vec();
        sorted_chunks.sort_by_key(|c| c.sequence_number);

        // Calculate total data size
        let total_size: usize = sorted_chunks.iter().map(|c| c.data.len()).sum();
        let mut result = Vec::with_capacity(total_size);

        // Concatenate chunks
        for chunk in sorted_chunks {
            result.extend_from_slice(&chunk.data);
        }

        Ok(result)
    }

    /// Validate processing chain
    pub fn validate_processing_chain(&chain: &network_message::ProcessingChain) -> Result<(), ProtocolError> {
        if chain.name.is_empty() {
            return Err(ProtocolError::InvalidProcessingChain("Empty name".to_string()));
        }

        if chain.steps.is_empty() {
            return Err(ProtocolError::InvalidProcessingChain("No processing steps".to_string()));
        }

        // Validate each step
        let mut used_orders = std::collections::HashSet::new();
        for (index, step) in chain.steps.iter().enumerate() {
            if step.id.is_empty() {
                return Err(ProtocolError::InvalidProcessingChain(
                    format!("Step {} has empty ID", index)
                ));
            }

            if step.r#type == 0 {
                return Err(ProtocolError::InvalidProcessingChain(
                    format!("Step {} has invalid type", index)
                ));
            }

            if used_orders.contains(&step.order) {
                return Err(ProtocolError::InvalidProcessingChain(
                    format!("Duplicate order {} in processing chain", step.order)
                ));
            }
            used_orders.insert(step.order);

            // Validate step parameters
            self.validate_step_parameters(step)?;
        }

        Ok(())
    }

    fn validate_step_parameters(&step: &network_message::ProcessingStep) -> Result<(), ProtocolError> {
        match step.r#type {
            1 => { // Equalizer
                let bands = step.parameters.get("bands")
                    .and_then(|v| v.parse::<i32>().ok())
                    .ok_or_else(|| ProtocolError::InvalidProcessingStep("Missing or invalid bands parameter".to_string()))?;

                if bands < 1 || bands > 512 {
                    return Err(ProtocolError::InvalidProcessingStep(
                        format!("Invalid number of equalizer bands: {}", bands)
                    ));
                }

                // Check for required parameters
                if !step.parameters.contains_key("frequencies") {
                    return Err(ProtocolError::InvalidProcessingStep(
                        "Missing frequencies parameter".to_string()
                    ));
                }

                if !step.parameters.contains_key("gains") {
                    return Err(ProtocolError::InvalidProcessingStep(
                        "Missing gains parameter".to_string()
                    ));
                }
            }
            2 => { // Convolution
                if !step.parameters.contains_key("impulse_file") {
                    return Err(ProtocolError::InvalidProcessingStep(
                        "Missing impulse_file parameter".to_string()
                    ));
                }

                if let Some(length_str) = step.parameters.get("length") {
                    if let Ok(length) = length_str.parse::<u64>() {
                        if length > 16_777_216 { // 16M points
                            return Err(ProtocolError::InvalidProcessingStep(
                                format!("Impulse response too long: {} points", length)
                            ));
                        }
                    }
                }
            }
            _ => {
                // Unknown step types are allowed for extensibility
            }
        }

        Ok(())
    }

    // Private helper methods
    fn write_header(&self, buffer: &mut Vec<u8>, header: &MessageHeader) -> Result<(), ProtocolError> {
        buffer.write_u8(header.version)?;
        buffer.write_u16::<LittleEndian>(header.message_type)?;
        buffer.write_u16::<LittleEndian>(header.flags)?;
        buffer.write_u32::<LittleEndian>(header.payload_size)?;
        buffer.write_u64::<LittleEndian>(header.timestamp)?;
        buffer.write_u32::<LittleEndian>(header.checksum)?;
        Ok(())
    }

    fn read_header(&self, data: &[u8]) -> Result<(MessageHeader, usize), ProtocolError> {
        let mut cursor = std::io::Cursor::new(data);

        let version = cursor.read_u8()?;
        let message_type = cursor.read_u16::<LittleEndian>()?;
        let flags = cursor.read_u16::<LittleEndian>()?;
        let payload_size = cursor.read_u32::<LittleEndian>()?;
        let timestamp = cursor.read_u64::<LittleEndian>()?;
        let checksum = cursor.read_u32::<LittleEndian>()?;

        Ok((
            MessageHeader {
                version,
                message_type,
                flags,
                payload_size,
                timestamp,
                checksum,
            },
            cursor.position() as usize,
        ))
    }

    fn validate_header(&self, header: &MessageHeader) -> Result<(), ProtocolError> {
        if header.version != 1 {
            return Err(ProtocolError::UnsupportedVersion(header.version));
        }

        if header.message_type == 0 {
            return Err(ProtocolError::InvalidMessageType(header.message_type));
        }

        // Check timestamp validity (within reasonable range)
        let current_time = self.current_timestamp();
        const MAX_TIME_DIFF: u64 = 3600_000_000; // 1 hour in microseconds
        if header.timestamp > current_time + MAX_TIME_DIFF ||
           (current_time > header.timestamp && current_time - header.timestamp > MAX_TIME_DIFF) {
            return Err(ProtocolError::InvalidTimestamp(header.timestamp));
        }

        Ok(())
    }

    fn calculate_checksum(&self, header: &MessageHeader, payload: &[u8]) -> u32 {
        let mut crc = 0xFFFFFFFF_u32;

        // Include header fields (except checksum itself)
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                header as *const _ as *const u8,
                std::mem::size_of::<MessageHeader>() - std::mem::size_of::<u32>(),
            )
        };

        for &byte in header_bytes {
            crc = self.update_crc32(crc, byte);
        }

        // Include payload
        for &byte in payload {
            crc = self.update_crc32(crc, byte);
        }

        !crc
    }

    fn calculate_crc32(&self, data: &[u8]) -> u32 {
        let mut crc = 0xFFFFFFFF_u32;
        for &byte in data {
            crc = self.update_crc32(crc, byte);
        }
        !crc
    }

    fn update_crc32(&self, crc: u32, byte: u8) -> u32 {
        const CRC_TABLE: [u32; 256] = [
            0x00000000, 0x77073096, 0xee0e612c, 0x990951ba,
            0x076dc419, 0x706af48f, 0xe963a535, 0x9e6495a3,
            0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
            0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91,
            // ... (full CRC32 table would be included in production)
        ];

        (crc >> 8) ^ CRC_TABLE[((crc ^ byte as u32) & 0xFF) as usize]
    }

    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, ProtocolError> {
        // Simple compression using miniz (or zlib in production)
        // For now, return uncompressed data as placeholder
        Ok(data.to_vec())
    }

    fn decompress_data(&self, compressed: &[u8]) -> Result<Vec<u8>, ProtocolError> {
        // Simple decompression
        // For now, return compressed data as placeholder
        Ok(compressed.to_vec())
    }

    fn current_timestamp(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64
    }
}

impl Default for BinaryProtocol {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct MessageHeader {
    pub version: u8,
    pub message_type: u16,
    pub flags: u16,
    pub payload_size: u32,
    pub timestamp: u64,
    pub checksum: u32,
}

#[derive(Debug, Clone)]
pub struct NetworkChunk {
    pub chunk_id: u32,
    pub total_chunks: u32,
    pub sequence_number: u32,
    pub data: Vec<u8>,
    pub checksum: u32,
}

#[derive(Debug, Clone)]
pub enum ProtocolError {
    InsufficientData,
    ChecksumMismatch,
    UnsupportedVersion(u8),
    InvalidMessageType(u16),
    InvalidTimestamp(u64),
    IncompleteChunks,
    ProtobufDecode(prost::DecodeError),
    InvalidProcessingChain(String),
    InvalidProcessingStep(String),
    CompressionError(String),
    DecompressionError(String),
    IoError(std::io::Error),
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProtocolError::InsufficientData => write!(f, "Insufficient data"),
            ProtocolError::ChecksumMismatch => write!(f, "Checksum mismatch"),
            ProtocolError::UnsupportedVersion(v) => write!(f, "Unsupported version: {}", v),
            ProtocolError::InvalidMessageType(t) => write!(f, "Invalid message type: {}", t),
            ProtocolError::InvalidTimestamp(t) => write!(f, "Invalid timestamp: {}", t),
            ProtocolError::IncompleteChunks => write!(f, "Incomplete chunks"),
            ProtocolError::ProtobufDecode(e) => write!(f, "Protobuf decode error: {}", e),
            ProtocolError::InvalidProcessingChain(s) => write!(f, "Invalid processing chain: {}", s),
            ProtocolError::InvalidProcessingStep(s) => write!(f, "Invalid processing step: {}", s),
            ProtocolError::CompressionError(s) => write!(f, "Compression error: {}", s),
            ProtocolError::DecompressionError(s) => write!(f, "Decompression error: {}", s),
            ProtocolError::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for ProtocolError {}

impl From<std::io::Error> for ProtocolError {
    fn from(error: std::io::Error) -> Self {
        ProtocolError::IoError(error)
    }
}

impl From<prost::DecodeError> for ProtocolError {
    fn from(error: prost::DecodeError) -> Self {
        ProtocolError::ProtobufDecode(error)
    }
}