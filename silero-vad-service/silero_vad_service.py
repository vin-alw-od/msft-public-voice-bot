#!/usr/bin/env python3
"""
Isolated Silero VAD Service
Runs as separate container, provides REST API for C# EchoBot integration.
Zero impact on existing Azure VAD pipeline.
"""

import torch
import numpy as np
import time
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import asyncio
from collections import deque
import io
import wave
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Silero VAD Service", 
    description="Isolated Voice Activity Detection service for Teams bot integration",
    version="1.0.0"
)

class VADConfig(BaseModel):
    """VAD configuration parameters"""
    threshold: float = Field(0.5, ge=0.1, le=0.9, description="VAD threshold (0.1-0.9)")
    min_speech_duration: int = Field(250, ge=100, le=2000, description="Min speech duration (ms)")
    min_silence_duration: int = Field(2000, ge=500, le=10000, description="Min silence duration (ms)")
    speech_pad_ms: int = Field(500, ge=100, le=1000, description="Speech padding (ms)")
    sample_rate: int = Field(16000, description="Audio sample rate (8000 or 16000)")

class VADRequest(BaseModel):
    """VAD detection request"""
    audio_data: str = Field(..., description="Base64 encoded audio data (16-bit PCM)")
    session_id: Optional[str] = Field(None, description="Session identifier for state tracking")
    config: Optional[VADConfig] = Field(None, description="Optional config override")

class VADResponse(BaseModel):
    """VAD detection response"""
    is_speech: bool = Field(..., description="Is speech detected")
    speech_probability: float = Field(..., description="Speech probability (0.0-1.0)")
    speech_start: Optional[float] = Field(None, description="Speech start timestamp")
    speech_end: Optional[float] = Field(None, description="Speech end timestamp") 
    speech_duration: Optional[float] = Field(None, description="Speech duration in ms")
    session_state: str = Field(..., description="Current session state")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class VADHealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    uptime_seconds: float
    total_requests: int
    average_processing_time_ms: float

class SileroVADProcessor:
    """Core Silero VAD processing engine"""
    
    def __init__(self):
        self.model = None
        self.utils = None
        self.config = VADConfig()
        self.is_loaded = False
        self.load_time = time.time()
        
        # Session state tracking
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.total_requests = 0
        self.total_processing_time = 0.0
        
        # Model will be loaded on startup event
    
    async def load_model(self):
        """Load Silero VAD model asynchronously"""
        try:
            logger.info("Loading Silero VAD model...")
            
            # Load model from torch hub
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False  # Use PyTorch version for better compatibility
            )
            
            # Set model to evaluation mode
            self.model.eval()
            
            self.is_loaded = True
            logger.info("✅ Silero VAD model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Silero VAD model: {e}")
            self.is_loaded = False
            raise
    
    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get or create session state"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'is_speech': False,
                'speech_start_time': None,
                'last_speech_time': None,
                'speech_segments': [],
                'created_at': time.time()
            }
        return self.sessions[session_id]
    
    def cleanup_old_sessions(self, max_age_hours: float = 2.0):
        """Clean up old sessions"""
        current_time = time.time()
        old_sessions = []
        
        for session_id, state in self.sessions.items():
            age_hours = (current_time - state['created_at']) / 3600
            if age_hours > max_age_hours:
                old_sessions.append(session_id)
        
        for session_id in old_sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up old session: {session_id}")
    
    async def detect_speech(self, audio_data: np.ndarray, session_id: str, config: VADConfig) -> VADResponse:
        """Process audio chunk and detect speech activity"""
        start_time = time.time()
        
        if not self.is_loaded:
            raise HTTPException(status_code=503, detail="VAD model not loaded")
        
        try:
            # Get session state
            session_state = self.get_session_state(session_id)
            current_timestamp = time.time()
            
            # Convert to float32 and normalize
            if audio_data.dtype != np.float32:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data
            
            # Ensure correct shape
            if len(audio_float.shape) > 1:
                audio_float = audio_float.flatten()
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_float)
            
            # Get VAD prediction
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, config.sample_rate).item()
            
            # Update speech state
            is_speech_detected = speech_prob > config.threshold
            speech_start = None
            speech_end = None
            speech_duration = None
            session_state_str = "listening"
            
            if is_speech_detected:
                if not session_state['is_speech']:
                    # Speech started
                    session_state['is_speech'] = True
                    session_state['speech_start_time'] = current_timestamp
                    speech_start = current_timestamp
                    session_state_str = "speech_started"
                    logger.debug(f"Speech started for session {session_id}")
                
                session_state['last_speech_time'] = current_timestamp
                session_state_str = "speech_active"
            
            else:
                if session_state['is_speech']:
                    # Check if silence duration is enough to end speech
                    silence_duration_ms = (current_timestamp - session_state['last_speech_time']) * 1000
                    
                    if silence_duration_ms > config.min_silence_duration:
                        # Speech ended
                        speech_duration = (current_timestamp - session_state['speech_start_time']) * 1000
                        
                        if speech_duration > config.min_speech_duration:
                            speech_end = current_timestamp
                            session_state['speech_segments'].append({
                                'start': session_state['speech_start_time'],
                                'end': current_timestamp,
                                'duration': speech_duration
                            })
                            session_state_str = "speech_ended"
                            logger.info(f"Speech ended for session {session_id}: {speech_duration:.0f}ms")
                        
                        session_state['is_speech'] = False
                        session_state['speech_start_time'] = None
                    else:
                        session_state_str = "speech_paused"
                else:
                    session_state_str = "silence"
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self.total_requests += 1
            self.total_processing_time += processing_time
            
            return VADResponse(
                is_speech=is_speech_detected,
                speech_probability=speech_prob,
                speech_start=speech_start,
                speech_end=speech_end,
                speech_duration=speech_duration,
                session_state=session_state_str,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"VAD processing error for session {session_id}: {e}")
            raise HTTPException(status_code=500, detail=f"VAD processing failed: {str(e)}")

# Global VAD processor instance
vad_processor = SileroVADProcessor()

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("🚀 Starting Silero VAD Service...")
    await vad_processor.load_model()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Silero VAD Service...")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "Silero VAD Service",
        "version": "1.0.0", 
        "status": "running",
        "model_loaded": vad_processor.is_loaded
    }

@app.get("/health", response_model=VADHealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - vad_processor.load_time
    avg_processing_time = (
        vad_processor.total_processing_time / vad_processor.total_requests 
        if vad_processor.total_requests > 0 else 0.0
    )
    
    return VADHealthResponse(
        status="healthy" if vad_processor.is_loaded else "unhealthy",
        model_loaded=vad_processor.is_loaded,
        uptime_seconds=uptime,
        total_requests=vad_processor.total_requests,
        average_processing_time_ms=avg_processing_time
    )

@app.post("/vad/detect", response_model=VADResponse)
async def detect_voice_activity(request: VADRequest):
    """Detect voice activity in audio data"""
    try:
        # Decode base64 audio data
        import base64
        audio_bytes = base64.b64decode(request.audio_data)
        
        # Convert bytes to numpy array (assuming 16-bit PCM)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Use provided config or default
        config = request.config or vad_processor.config
        
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{int(time.time() * 1000)}"
        
        # Process audio
        result = await vad_processor.detect_speech(audio_np, session_id, config)
        
        return result
        
    except Exception as e:
        logger.error(f"VAD detection error: {e}")
        raise HTTPException(status_code=500, detail=f"VAD detection failed: {str(e)}")

@app.post("/vad/detect-file")
async def detect_voice_activity_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
    threshold: float = 0.5
):
    """Detect voice activity from uploaded audio file"""
    try:
        # Read file content
        content = await file.read()
        
        # Handle WAV files
        if file.filename.lower().endswith('.wav'):
            with wave.open(io.BytesIO(content), 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(-1)
                audio_np = np.frombuffer(frames, dtype=np.int16)
        else:
            # Assume raw PCM data
            audio_np = np.frombuffer(content, dtype=np.int16)
            sample_rate = 16000
        
        # Create config
        config = VADConfig(
            threshold=threshold,
            sample_rate=sample_rate
        )
        
        # Generate session ID if not provided
        session_id = session_id or f"file_session_{int(time.time() * 1000)}"
        
        # Process audio
        result = await vad_processor.detect_speech(audio_np, session_id, config)
        
        return result
        
    except Exception as e:
        logger.error(f"File VAD detection error: {e}")
        raise HTTPException(status_code=500, detail=f"File VAD detection failed: {str(e)}")

@app.post("/vad/config", response_model=VADConfig)
async def update_vad_config(config: VADConfig):
    """Update VAD configuration"""
    try:
        vad_processor.config = config
        logger.info(f"VAD config updated: {config}")
        return config
    except Exception as e:
        logger.error(f"Config update error: {e}")
        raise HTTPException(status_code=500, detail=f"Config update failed: {str(e)}")

@app.get("/vad/config", response_model=VADConfig)
async def get_vad_config():
    """Get current VAD configuration"""
    return vad_processor.config

@app.get("/vad/sessions")
async def list_sessions():
    """List active sessions (for debugging)"""
    return {
        "total_sessions": len(vad_processor.sessions),
        "sessions": {
            session_id: {
                "is_speech": state['is_speech'],
                "segments_count": len(state['speech_segments']),
                "created_at": datetime.fromtimestamp(state['created_at']).isoformat()
            }
            for session_id, state in vad_processor.sessions.items()
        }
    }

@app.delete("/vad/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    if session_id in vad_processor.sessions:
        del vad_processor.sessions[session_id]
        return {"message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.post("/vad/sessions/cleanup")
async def cleanup_sessions(max_age_hours: float = 2.0):
    """Cleanup old sessions"""
    initial_count = len(vad_processor.sessions)
    vad_processor.cleanup_old_sessions(max_age_hours)
    final_count = len(vad_processor.sessions)
    
    return {
        "message": f"Cleaned up {initial_count - final_count} sessions",
        "remaining_sessions": final_count
    }

# Background task for periodic cleanup
@app.on_event("startup")
async def start_background_tasks():
    """Start background cleanup task"""
    async def periodic_cleanup():
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                vad_processor.cleanup_old_sessions()
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
    
    asyncio.create_task(periodic_cleanup())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,  # Different port from main survey API
        reload=True
    )