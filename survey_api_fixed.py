from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
from typing import Dict, Any, Optional, List
import json
from datetime import datetime, timedelta
import os
import asyncio
from contextlib import asynccontextmanager

# Import the fixed SurveyAgent
from ConvSurveyAgent_Fixed import ConversationSession

app = FastAPI(title="Survey Agent API", description="LLM-powered conversational survey API for Teams bot integration")

# Global session storage with proper cleanup
active_sessions: Dict[str, ConversationSession] = {}

# Pydantic models for API requests/responses
class StartSurveyRequest(BaseModel):
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class StartSurveyResponse(BaseModel):
    session_id: str
    message: str
    status: str

class ProcessInputRequest(BaseModel):
    session_id: str
    user_input: str

class ProcessInputResponse(BaseModel):
    session_id: str
    message: str
    status: str  # "collecting", "follow_up", "completed", "error"
    collected_data: Optional[Dict[str, Any]] = None
    missing_fields: Optional[List[str]] = None

class SessionStatusResponse(BaseModel):
    session_id: str
    status: str
    collected_data: Dict[str, Any]
    missing_fields: List[str]


# Background task for session cleanup
async def cleanup_expired_sessions():
    """Clean up expired sessions periodically."""
    while True:
        try:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session in active_sessions.items():
                if session.is_expired(timeout_hours=2):  # 2 hour timeout
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                session = active_sessions[session_id]
                
                # Save CSV data before cleanup if session has collected data
                try:
                    if hasattr(session, 'agent') and session.agent.collected_data:
                        collected_fields = sum(1 for v in session.agent.collected_data.values() if v)
                        if collected_fields > 0:
                            print(f"Saving CSV data for expired session {session_id} ({collected_fields} fields collected)")
                            session.agent.update_initiatives_csv(session.agent.collected_data.copy())
                            print(f"CSV data saved for expired session {session_id}")
                except Exception as e:
                    print(f"Failed to save CSV data for expired session {session_id}: {e}")
                
                del active_sessions[session_id]
                print(f"Cleaned up expired session: {session_id}")
            
            if expired_sessions:
                print(f"Cleaned up {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            print(f"Error in session cleanup: {e}")
        
        # Run cleanup every 30 minutes
        await asyncio.sleep(1800)


# Lifespan context manager for background tasks
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background tasks
    cleanup_task = asyncio.create_task(cleanup_expired_sessions())
    
    yield
    
    # Clean up on shutdown
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


# Update FastAPI app with lifespan
app = FastAPI(
    title="Survey Agent API",
    description="LLM-powered conversational survey API for Teams bot integration",
    lifespan=lifespan
)


# API Endpoints
@app.get("/")
async def root():
    return {"message": "Survey Agent API is running", "version": "2.0.0"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(active_sessions)
    }


@app.post("/start-survey", response_model=StartSurveyResponse)
async def start_survey(request: StartSurveyRequest):
    """Start a new survey session with improved error handling."""
    try:
        session_id = str(uuid.uuid4())
        
        # Create new conversation session
        session = ConversationSession(session_id, request.user_id)
        
        # Get initial greeting
        initial_message = session.get_initial_greeting()
        
        # Store session
        active_sessions[session_id] = session
        
        print(f"Created new session {session_id} for user {request.user_id}")
        
        return StartSurveyResponse(
            session_id=session_id,
            message=initial_message,
            status="active"
        )
        
    except Exception as e:
        print(f"Error starting survey: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting survey: {str(e)}")


@app.post("/process-input", response_model=ProcessInputResponse) 
async def process_input(request: ProcessInputRequest):
    """Process user input for an existing session with improved session management."""
    try:
        # Check if session exists
        if request.session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[request.session_id]
        
        # Check if session is still active
        if session.status == "completed":
            raise HTTPException(status_code=400, detail="Session already completed")
        
        if session.status == "error":
            raise HTTPException(status_code=400, detail="Session is in error state")
        
        # Allow continued conversation in follow_up mode
        # session.status can be "active" or "follow_up" - both allow processing
        
        # Process the input with timeout protection
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(session.process_input, request.user_input),
                timeout=90.0  # 90 second timeout to match LLMService.cs
            )
        except asyncio.TimeoutError:
            session.status = "error"
            raise HTTPException(status_code=408, detail="Processing timed out")
        
        # Handle session completion - DON'T auto-delete
        if result["status"] == "completed":
            print(f"Session {request.session_id} completed successfully")
            # Keep session for a while in case client needs to access final data
            # It will be cleaned up by the background task later
        
        return ProcessInputResponse(
            session_id=request.session_id,
            message=result["message"],
            status=result["status"],
            collected_data=result.get("collected_data"),
            missing_fields=result.get("missing_fields")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing input for session {request.session_id}: {e}")
        
        # Try to update session status if it exists
        if request.session_id in active_sessions:
            active_sessions[request.session_id].status = "error"
        
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")


@app.get("/session/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """Get the current status of a session."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    status_data = session.get_status()
    
    return SessionStatusResponse(
        session_id=session_id,
        status=status_data["status"],
        collected_data=status_data["collected_data"],
        missing_fields=status_data["missing_fields"]
    )


@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    """End a session and clean up resources."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get session before deletion
    session = active_sessions[session_id]
    final_status = session.get_status()
    
    # Save CSV data and conversation log before deleting session
    try:
        # Save CSV data if session has collected data
        if hasattr(session, 'agent') and session.agent.collected_data:
            collected_fields = sum(1 for v in session.agent.collected_data.values() if v)
            if collected_fields > 0:
                print(f"Saving CSV data for manually ended session {session_id} ({collected_fields} fields collected)")
                session.agent.update_initiatives_csv(session.agent.collected_data.copy())
                print(f"CSV data saved for manually ended session {session_id}")
    except Exception as e:
        print(f"Failed to save CSV data for manually ended session {session_id}: {e}")
    
    try:
        from conversation_logger import log_conversation_event, save_conversation
        
        # Log manual session termination
        log_conversation_event(session_id, "session_manually_ended", {
            "termination_reason": "manual_deletion",
            "final_status": session.status,
            "collected_data": session.agent.collected_data
        })
        
        # Save the complete conversation log
        print(f"Saving conversation log for manually ended session {session_id}")
        save_conversation(session_id, session.user_id)
        print(f"Conversation log saved for manually ended session {session_id}")
        
    except Exception as e:
        print(f"Failed to save conversation log for manually ended session {session_id}: {e}")
    
    # Delete session
    del active_sessions[session_id]
    
    print(f"Manually ended session {session_id}")
    
    return {
        "message": "Session ended successfully",
        "final_status": final_status
    }


@app.get("/sessions")
async def list_active_sessions():
    """List all active sessions (for debugging and monitoring)."""
    sessions_info = []
    
    for session_id, session in active_sessions.items():
        status_data = session.get_status()
        sessions_info.append({
            "session_id": session_id,
            "user_id": session.user_id,
            "status": status_data["status"],
            "created_at": status_data["created_at"],
            "last_activity": status_data["last_activity"],
            "fields_collected": len([v for v in status_data["collected_data"].values() if v]),
            "total_fields": len(status_data["collected_data"])
        })
    
    return {
        "total_sessions": len(active_sessions),
        "sessions": sessions_info
    }


@app.get("/sessions/cleanup")
async def manual_cleanup():
    """Manually trigger session cleanup (for debugging)."""
    expired_count = 0
    expired_sessions = []
    
    for session_id, session in list(active_sessions.items()):
        if session.is_expired(timeout_hours=2):
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del active_sessions[session_id]
        expired_count += 1
    
    return {
        "message": f"Cleaned up {expired_count} expired sessions",
        "cleaned_sessions": expired_sessions,
        "remaining_sessions": len(active_sessions)
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors."""
    print(f"Unexpected error: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)