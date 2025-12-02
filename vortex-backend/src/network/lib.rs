//! Vortex Network Services
//!
//! This crate provides Rust-based network services for the Vortex GPU Audio Backend,
//! including HTTP API, WebSocket real-time streaming, and device discovery.

pub mod http_server;
pub mod websocket_server;
pub mod discovery_service;
pub mod protocol;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}