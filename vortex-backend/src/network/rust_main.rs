use anyhow::Result;
use std::env;
use tracing::{info, error};
use tracing_subscriber::{layer::SubscriberExt, EnvFilter};

// Import our modules
use vortex_network::create_http_server;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    info!("Vortex GPU Audio Backend Network Services Starting");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        return Ok(());
    }

    match args[1].as_str() {
        "http" => {
            info!("Starting HTTP server...");
            start_http_server().await
        }
        "websocket" => {
            info!("Starting WebSocket server...");
            start_websocket_server().await
        }
        "discovery" => {
            info!("Starting device discovery service...");
            start_discovery_service().await
        }
        _ => {
            error!("Unknown service: {}", args[1]);
            print_usage(&args[0]);
            Ok(())
        }
    }
}

async fn start_http_server() -> Result<()> {
    let app = create_http_server().await?;

    let bind_addr = "0.0.0.0:8080";
    info!("HTTP server listening on: {}", bind_addr);

    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn start_websocket_server() -> Result<()> {
    // TODO: Implement WebSocket server
    info!("WebSocket server implementation not yet complete");
    Ok(())
}

async fn start_discovery_service() -> Result<()> {
    // TODO: Implement device discovery
    info!("Device discovery service implementation not yet complete");
    Ok(())
}

fn print_usage(program_name: &str) {
    println!("Usage: {} <service>", program_name);
    println!("  Available services:");
    println!("    http      - HTTP REST API server");
    println!("    websocket - WebSocket real-time server");
    println!("    discovery - Device discovery service");
}