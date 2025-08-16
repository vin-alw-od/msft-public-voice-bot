from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import os

# Import your existing SurveyAgent class
from ConvSuveyAgentLatest_WorkingCopy import SurveyAgent
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="Survey Agent API", description="LLM-powered conversational survey API for Teams bot integration")

# Global session storage (in production, use Redis or database)
active_sessions: Dict[str, Dict[str, Any]] = {}

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
    status: str  # "collecting", "completed", "error"
    collected_data: Optional[Dict[str, Any]] = None
    missing_fields: Optional[List[str]] = None

class SessionStatusResponse(BaseModel):
    session_id: str
    status: str
    collected_data: Dict[str, Any]
    missing_fields: List[str]

# Helper class to manage survey sessions
class SurveySession:
    def __init__(self, session_id: str, user_id: str = None):
        self.session_id = session_id
        self.user_id = user_id
        self.agent = SurveyAgent()
        self.messages = []
        self.created_at = datetime.now()
        self.is_first_turn = True
        self.current_status = "active"  # active, completed, error
        
    def get_initial_greeting(self):
        """Get the initial greeting from the agent"""
        try:
            # Format the first prompt with empty history
            current_status = self.agent.format_current_status()
            missing = self.agent.get_missing_fields()
            
            first_prompt = self.agent.chat_prompt.format_messages(
                initial_context=self.agent.initial_context,
                current_status=current_status,
                missing_fields=", ".join(missing),
                history=[],
                input="",  # Empty input for first turn
                previous_initiatives_context=self.agent.previous_initiatives_context
            )
            
            response = self.agent.llm.invoke(first_prompt)
            assistant_msg = response.content.strip()
            
            # Extract just the answer part (after "Answer:")
            if "Answer:" in assistant_msg:
                answer = assistant_msg.split("Answer:")[-1].strip()
            else:
                answer = assistant_msg
            
            # Quick fix: If we get a generic greeting, replace with proper survey greeting
            if answer.strip() in ["Hello! How can I assist you today?", "Hello! How can I assist you today with your AI initiatives?", "Hello! How can I help you today?"]:
                answer = "Hello! I'd love to learn more about the AI initiatives you're currently working on at Hewlett Packard Enterprise. Can you tell me about the specific AI project you're currently involved in?"
                
            # Store the full response in messages for context
            self.messages.extend(first_prompt)
            self.messages.append(AIMessage(content=assistant_msg))
            
            self.is_first_turn = False
            return answer
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating initial greeting: {str(e)}")
    
    def process_user_input(self, user_input: str):
        """Process user input and return agent response"""
        try:
            # Add timeout protection to prevent infinite processing
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Processing timed out after 25 seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(25)  # 25 second timeout
            # Check for exit conditions
            if user_input.lower() in ("no", "n", "bye", "exit", "quit"):
                self.current_status = "completed"
                return {
                    "message": "Thank you for your time. Have a great day!",
                    "status": "completed",
                    "collected_data": self.agent.collected_data,
                    "missing_fields": []
                }
            
            # Get the last assistant message for extraction
            last_assistant_msg = ""
            for msg in reversed(self.messages):
                if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == "ai":
                    last_assistant_msg = msg.content
                    break
            
            # Extract fields from the conversation
            if last_assistant_msg:
                conversation_snippet = f"Assistant: {last_assistant_msg}\nUser: {user_input}"
                self.agent.extract_all_fields_working_copy(conversation_snippet)
            
            # Check if everything is collected
            missing = self.agent.get_missing_fields()
            if not missing:
                # All fields collected
                self.current_status = "completed"
                # Update CSV with new initiative
                self.agent.update_initiatives_csv(self.agent.collected_data.copy())
                return {
                    "message": "Thank you! I've collected all the information about your AI initiative.",
                    "status": "completed", 
                    "collected_data": self.agent.collected_data,
                    "missing_fields": []
                }
            
            # Generate next response
            current_status = self.agent.format_current_status()
            
            # Use full message history (no filtering needed since we maintain LangChain format)
            next_prompt = self.agent.chat_prompt.format_messages(
                initial_context=self.agent.initial_context,
                current_status=current_status,
                missing_fields=", ".join(missing),
                history=self.messages,
                input=user_input,
                previous_initiatives_context=self.agent.previous_initiatives_context
            )
            
            response = self.agent.llm.invoke(next_prompt)
            assistant_msg = response.content.strip()
            
            # Extract just the answer part
            if "Answer:" in assistant_msg:
                answer = assistant_msg.split("Answer:")[-1].strip()
            elif "Thought 5:" in assistant_msg:
                # Fallback: extract the question from Thought 5 if no Answer section
                thought5_section = assistant_msg.split("Thought 5:")[-1].strip()
                # Look for a question at the end of Thought 5
                if "?" in thought5_section:
                    # Extract the last sentence that ends with a question mark
                    sentences = thought5_section.split(".")
                    for sentence in reversed(sentences):
                        if "?" in sentence:
                            answer = sentence.strip()
                            break
                    else:
                        answer = thought5_section
                else:
                    answer = thought5_section
            else:
                answer = assistant_msg
            
            # Update message history
            self.messages.append(HumanMessage(content=user_input))
            self.messages.append(AIMessage(content=assistant_msg))
            
            result = {
                "message": answer,
                "status": "collecting",
                "collected_data": self.agent.collected_data,
                "missing_fields": missing
            }
            
            # Clear the timeout
            signal.alarm(0)
            return result
            
        except TimeoutError as e:
            signal.alarm(0)  # Clear timeout
            self.current_status = "error"
            raise HTTPException(status_code=408, detail="Processing timed out - please try again")
        except Exception as e:
            signal.alarm(0)  # Clear timeout
            self.current_status = "error"
            raise HTTPException(status_code=500, detail=f"Error processing user input: {str(e)}")

# API Endpoints

@app.get("/")
async def root():
    return {"message": "Survey Agent API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/start-survey", response_model=StartSurveyResponse)
async def start_survey(request: StartSurveyRequest):
    """Start a new survey session"""
    try:
        session_id = str(uuid.uuid4())
        session = SurveySession(session_id, request.user_id)
        
        # Get initial greeting
        initial_message = session.get_initial_greeting()
        
        # Store session
        active_sessions[session_id] = {
            "session": session,
            "created_at": datetime.now().isoformat()
        }
        
        return StartSurveyResponse(
            session_id=session_id,
            message=initial_message,
            status="active"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting survey: {str(e)}")

@app.post("/process-input", response_model=ProcessInputResponse) 
async def process_input(request: ProcessInputRequest):
    """Process user input for an existing session"""
    try:
        if request.session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[request.session_id]["session"]
        
        if session.current_status == "completed":
            raise HTTPException(status_code=400, detail="Session already completed")
        
        # Process the input
        result = session.process_user_input(request.user_input)
        
        # Clean up completed sessions
        if result["status"] == "completed":
            del active_sessions[request.session_id]
        
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
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")

@app.get("/session/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """Get the current status of a session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]["session"]
    
    return SessionStatusResponse(
        session_id=session_id,
        status=session.current_status,
        collected_data=session.agent.collected_data,
        missing_fields=session.agent.get_missing_fields()
    )

@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    """End a session and clean up resources"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del active_sessions[session_id]
    return {"message": "Session ended successfully"}

@app.get("/sessions")
async def list_active_sessions():
    """List all active sessions (for debugging)"""
    return {
        "active_sessions": len(active_sessions),
        "sessions": [
            {
                "session_id": sid,
                "created_at": data["created_at"],
                "status": data["session"].current_status
            }
            for sid, data in active_sessions.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)