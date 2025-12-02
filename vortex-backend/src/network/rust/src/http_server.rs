use axum::{
    extract::{Multipart, Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::fs;
use tower::ServiceBuilder;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::{error, info, warn};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct AppState {
    pub upload_dir: PathBuf,
    pub files: Arc<RwLock<HashMap<String, FileInfo>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub id: String,
    pub filename: String,
    pub content_type: String,
    pub size: u64,
    pub uploaded_at: u64,
    pub status: FileStatus,
    pub metadata: Option<AudioMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileStatus {
    Uploading,
    Processing,
    Ready,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMetadata {
    pub format: String,
    pub codec: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub bit_depth: u8,
    pub duration: Option<f64>,
    pub bitrate: Option<u32>,
    pub title: Option<String>,
    pub artist: Option<String>,
    pub album: Option<String>,
    pub genre: Option<String>,
    pub year: Option<u16>,
    pub track: Option<u16>,
}

#[derive(Debug, Serialize)]
pub struct UploadResponse {
    pub success: bool,
    pub file_id: Option<String>,
    pub message: String,
    pub file_info: Option<FileInfo>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub success: false,
    pub error: String,
    pub code: u16,
}

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> Response {
        let status = match self.code {
            400 => StatusCode::BAD_REQUEST,
            404 => StatusCode::NOT_FOUND,
            413 => StatusCode::PAYLOAD_TOO_LARGE,
            415 => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            500 => StatusCode::INTERNAL_SERVER_ERROR,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };

        (status, Json(self)).into_response()
    }
}

#[derive(Debug, Deserialize)]
pub struct AudioFormatsQuery {
    pub supported: Option<bool>,
    pub container: Option<String>,
}

pub async fn create_http_server() -> anyhow::Result<Router> {
    let upload_dir = PathBuf::from("uploads");

    // Create upload directory if it doesn't exist
    if !upload_dir.exists() {
        fs::create_dir_all(&upload_dir).await?;
    }

    let state = AppState {
        upload_dir,
        files: Arc::new(RwLock::new(HashMap::new())),
    };

    let app = Router::new()
        // Audio file upload endpoint
        .route("/api/audio/upload", post(upload_audio_file))
        // Audio files list endpoint
        .route("/api/audio/files", get(list_audio_files))
        // Audio file info endpoint
        .route("/api/audio/files/:id", get(get_file_info))
        // Audio file delete endpoint
        .route("/api/audio/files/:id", axum::routing::delete(delete_file))
        // Audio formats endpoint
        .route("/api/audio/formats", get(get_audio_formats))
        // Audio metadata endpoint
        .route("/api/audio/files/:id/metadata", get(get_file_metadata))
        // Audio status endpoint
        .route("/api/audio/status", get(get_processing_status))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(
                    CorsLayer::new()
                        .allow_origin(Any)
                        .allow_methods(Any)
                        .allow_headers(Any),
                ),
        )
        .with_state(state);

    Ok(app)
}

/// Upload audio file endpoint
pub async fn upload_audio_file(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, ErrorResponse> {
    info!("Received audio file upload request");

    let mut uploaded_files = Vec::new();

    while let Some(field) = multipart.next_field().await.map_err(|e| ErrorResponse {
        success: false,
        error: format!("Failed to read multipart field: {}", e),
        code: 400,
    })? {
        let name = field.name().unwrap_or("file");
        let filename = field.file_name().unwrap_or("unknown");
        let content_type = field.content_type().unwrap_or("application/octet-stream");

        // Validate file type
        if !is_supported_audio_type(content_type) && !is_supported_audio_extension(filename) {
            warn!("Unsupported file type: {} (filename: {})", content_type, filename);
            continue;
        }

        // Generate unique file ID
        let file_id = Uuid::new_v4().to_string();

        // Create file path
        let file_extension = PathBuf::from(filename)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("bin");
        let file_path = state.upload_dir.join(format!("{}.{}", file_id, file_extension));

        // Get file data
        let file_data = field.bytes().await.map_err(|e| ErrorResponse {
            success: false,
            error: format!("Failed to read file data: {}", e),
            code: 400,
        })?;

        let file_size = file_data.len();

        // Check file size limit (2GB)
        if file_size > 2_147_483_648 {
            return Err(ErrorResponse {
                success: false,
                error: "File too large. Maximum size is 2GB".to_string(),
                code: 413,
            });
        }

        // Save file to disk
        fs::write(&file_path, &file_data).await.map_err(|e| ErrorResponse {
            success: false,
            error: format!("Failed to save file: {}", e),
            code: 500,
        })?;

        // Create file info
        let file_info = FileInfo {
            id: file_id.clone(),
            filename: filename.to_string(),
            content_type: content_type.to_string(),
            size: file_size as u64,
            uploaded_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            status: FileStatus::Processing,
            metadata: None, // Will be populated by processing task
        };

        // Store file info
        state.files.write().unwrap().insert(file_id.clone(), file_info.clone());

        // Start processing in background
        let file_info_clone = file_info.clone();
        let state_clone = state.clone();
        tokio::spawn(async move {
            if let Err(e) = process_audio_file(state_clone, file_info_clone).await {
                error!("Failed to process audio file {}: {}", file_info_clone.id, e);
            }
        });

        uploaded_files.push(file_info);
        info!("Successfully uploaded file: {} (ID: {})", filename, file_id);
    }

    if uploaded_files.is_empty() {
        return Err(ErrorResponse {
            success: false,
            error: "No valid audio files found in upload".to_string(),
            code: 400,
        });
    }

    let response = UploadResponse {
        success: true,
        file_id: Some(uploaded_files[0].id.clone()),
        message: format!("Successfully uploaded {} file(s)", uploaded_files.len()),
        file_info: Some(uploaded_files[0].clone()),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// List uploaded audio files
pub async fn list_audio_files(
    State(state): State<AppState>,
) -> impl IntoResponse {
    let files = state.files.read().unwrap();
    let file_list: Vec<FileInfo> = files.values().cloned().collect();

    Json(file_list)
}

/// Get file information
pub async fn get_file_info(
    State(state): State<AppState>,
    Path(file_id): Path<String>,
) -> Result<impl IntoResponse, ErrorResponse> {
    let files = state.files.read().unwrap();

    match files.get(&file_id) {
        Some(file_info) => Ok(Json(file_info.clone())),
        None => Err(ErrorResponse {
            success: false,
            error: "File not found".to_string(),
            code: 404,
        }),
    }
}

/// Delete uploaded file
pub async fn delete_file(
    State(state): State<AppState>,
    Path(file_id): Path<String>,
) -> Result<impl IntoResponse, ErrorResponse> {
    let mut files = state.files.write().unwrap();

    match files.remove(&file_id) {
        Some(file_info) => {
            // Remove file from disk
            let file_path = state.upload_dir.join(format!("{}.{}",
                file_info.id,
                PathBuf::from(&file_info.filename)
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("bin")
            ));

            if let Err(e) = fs::remove_file(file_path).await {
                warn!("Failed to remove file from disk: {}", e);
            }

            Ok(Json(UploadResponse {
                success: true,
                file_id: Some(file_id),
                message: "File deleted successfully".to_string(),
                file_info: None,
            }))
        }
        None => Err(ErrorResponse {
            success: false,
            error: "File not found".to_string(),
            code: 404,
        }),
    }
}

/// Get supported audio formats
pub async fn get_audio_formats(
    _query: Option<Json<AudioFormatsQuery>>,
) -> impl IntoResponse {
    let formats = serde_json::json!({
        "supported_formats": [
            {
                "extension": "wav",
                "mime_type": "audio/wav",
                "description": "Waveform Audio File Format",
                "codecs": ["PCM"],
                "max_sample_rate": 384000,
                "max_bit_depth": 32,
                "max_channels": 8
            },
            {
                "extension": "flac",
                "mime_type": "audio/flac",
                "description": "Free Lossless Audio Codec",
                "codecs": ["FLAC"],
                "max_sample_rate": 384000,
                "max_bit_depth": 24,
                "max_channels": 8
            },
            {
                "extension": "dsf",
                "mime_type": "audio/x-dsf",
                "description": "DSD Stream File",
                "codecs": ["DSD64", "DSD128", "DSD256", "DSD512", "DSD1024"],
                "max_sample_rate": 45158400,
                "max_bit_depth": 1,
                "max_channels": 6
            },
            {
                "extension": "dff",
                "mime_type": "audio/x-dff",
                "description": "DSD Interchange File Format",
                "codecs": ["DSD64", "DSD128", "DSD256", "DSD512", "DSD1024"],
                "max_sample_rate": 45158400,
                "max_bit_depth": 1,
                "max_channels": 6
            }
        ],
        "container_formats": {
            "dsf": "DSD Stream File container for DSD audio",
            "dff": "DSDIFF container for high-resolution DSD audio"
        },
        "max_file_size": 2147483648,
        "processing": {
            "gpu_acceleration": true,
            "real_time": false,
            "max_concurrent_files": 4
        }
    });

    Json(formats)
}

/// Get file metadata
pub async fn get_file_metadata(
    State(state): State<AppState>,
    Path(file_id): Path<String>,
) -> Result<impl IntoResponse, ErrorResponse> {
    let files = state.files.read().unwrap();

    match files.get(&file_id) {
        Some(file_info) => {
            match &file_info.metadata {
                Some(metadata) => Ok(Json(metadata)),
                None => Err(ErrorResponse {
                    success: false,
                    error: "Metadata not yet processed".to_string(),
                    code: 404,
                }),
            }
        }
        None => Err(ErrorResponse {
            success: false,
            error: "File not found".to_string(),
            code: 404,
        }),
    }
}

/// Get processing status
pub async fn get_processing_status(
    State(state): State<AppState>,
) -> impl IntoResponse {
    let files = state.files.read().unwrap();
    let total_files = files.len();

    let status_counts = files.values()
        .fold(HashMap::new(), |mut acc, file| {
            *acc.entry(format!("{:?}", file.status)).or_insert(0) += 1;
            acc
        });

    let status = serde_json::json!({
        "total_files": total_files,
        "status_counts": status_counts,
        "gpu_available": true, // TODO: Check GPU availability
        "max_concurrent_processing": 4,
        "current_processing": status_counts.get("Processing").unwrap_or(&0),
        "ready_for_processing": status_counts.get("Uploading").unwrap_or(&0)
    });

    Json(status)
}

// Helper functions

fn is_supported_audio_type(content_type: &str) -> bool {
    matches!(content_type.to_lowercase().as_str(),
        "audio/wav" | "audio/wave" | "audio/x-wav" |
        "audio/flac" | "audio/x-flac" |
        "audio/x-dsf" | "audio/x-dff" |
        "audio/dsf" | "audio/dff" |
        "application/octet-stream" // We'll validate by extension
    )
}

fn is_supported_audio_extension(filename: &str) -> bool {
    if let Some(extension) = PathBuf::from(filename).extension() {
        if let Some(ext_str) = extension.to_str() {
            matches!(ext_str.to_lowercase().as_str(),
                "wav" | "wave" | "flac" | "dsf" | "dff"
            )
        } else {
            false
        }
    } else {
        false
    }
}

async fn process_audio_file(
    state: AppState,
    file_info: FileInfo,
) -> anyhow::Result<()> {
    info!("Processing audio file: {}", file_info.id);

    // Update status to processing
    {
        let mut files = state.files.write().unwrap();
        if let Some(info) = files.get_mut(&file_info.id) {
            info.status = FileStatus::Processing;
        }
    }

    // TODO: Implement actual audio processing using C++ backend
    // For now, simulate processing with a delay
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Simulate extracted metadata
    let metadata = AudioMetadata {
        format: file_info.content_type.clone(),
        codec: "PCM".to_string(), // This would be detected
        sample_rate: 44100,
        channels: 2,
        bit_depth: 16,
        duration: Some(180.5), // 3 minutes
        bitrate: Some(1411),
        title: Some("Sample Title".to_string()),
        artist: Some("Sample Artist".to_string()),
        album: Some("Sample Album".to_string()),
        genre: Some("Sample Genre".to_string()),
        year: Some(2023),
        track: Some(1),
    };

    // Update file info with metadata and status
    {
        let mut files = state.files.write().unwrap();
        if let Some(info) = files.get_mut(&file_info.id) {
            info.metadata = Some(metadata);
            info.status = FileStatus::Ready;
        }
    }

    info!("Successfully processed audio file: {}", file_info.id);
    Ok(())
}