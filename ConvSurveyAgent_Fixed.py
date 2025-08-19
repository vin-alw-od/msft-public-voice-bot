import csv
import json
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
import re
import requests
from io import StringIO
from typing import Dict, List, Any, Optional
import asyncio
import threading
from datetime import datetime, timedelta
from conversation_logger import log_bot_question, log_user_answer, log_conversation_event, save_conversation

# Configuration for logging features
ENABLE_LATENCY_LOGGING = os.getenv("ENABLE_LATENCY_LOGGING", "true").lower() == "true"
ENABLE_EXTRACTION_FAILURE_LOGGING = os.getenv("ENABLE_EXTRACTION_FAILURE_LOGGING", "true").lower() == "true"
MAX_LOGGING_LATENCY_MS = float(os.getenv("MAX_LOGGING_LATENCY_MS", "50"))  # Disable logging if it takes too long

def safe_log_event(session_id: str, event: str, data: Dict, max_time_ms: float = MAX_LOGGING_LATENCY_MS):
    """Safely log events with performance budget and proper error handling."""
    if not ENABLE_LATENCY_LOGGING and "latency" in event:
        return
    if not ENABLE_EXTRACTION_FAILURE_LOGGING and "failure" in event:
        return
        
    log_start = datetime.now()
    try:
        log_conversation_event(session_id, event, data)
    except Exception as e:
        # Log to console but don't fail the main operation
        print(f"WARNING: Logging failed for {event}: {e}")
    finally:
        # Check if logging took too long
        log_duration = (datetime.now() - log_start).total_seconds() * 1000
        if log_duration > max_time_ms:
            print(f"WARNING: Logging {event} took {log_duration:.1f}ms (max: {max_time_ms}ms)")


class SurveyAgent:
    """Enhanced SurveyAgent with proper LangChain implementation and conversation management."""
    
    def __init__(self):
        # Initialize LLM with better settings
        self.llm = ChatOpenAI(
            temperature=0.3,  # Increased for faster inference
            model="gpt-4o-mini",  # Use GPT-4 mini for higher rate limits
            max_tokens=150,  # Reduced from 1000 - bot responses are typically 20-80 tokens
            timeout=90  # Match LLMService.cs timeout
        )
        
        # Load configuration and data
        self._load_configuration()
        self._load_context_data()
        self._load_slots_data()
        self._setup_prompts()
        
        # Initialize data structures
        self.collected_data = {field: None for field in self.slots}
        self.previous_initiatives = self._load_previous_initiatives()
        self.conversation_history = []
        self.max_conversation_length = 50  # Prevent memory explosion
        self.in_follow_up_mode = False  # Track if we're in follow-up conversation mode
        self.additional_initiatives = []  # Track additional initiatives mentioned in follow-up
        self.collecting_additional = False  # Track if we're collecting data for additional initiative
        self.current_additional_data = {}  # Data for current additional initiative being collected
        
    def _load_configuration(self):
        """Load Azure storage configuration."""
        self.sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN", "")
        self.base_url = "https://acu1tmagentbotd1saglobal.blob.core.windows.net/csvdata"
        
    def _load_context_data(self):
        """Load initial context from CSV."""
        try:
            context_url = self._build_url("Test_Context_HPE.csv")
            print(f"Loading context from: {context_url}")
            
            response = requests.get(context_url, timeout=30)
            response.raise_for_status()
            
            context_csv = StringIO(response.text)
            reader = csv.DictReader(context_csv)
            context_row = next(reader)
            
            self.initial_context = self._format_context(context_row)
            print(f"Successfully loaded context: {len(response.text)} chars")
            
        except Exception as e:
            print(f"Error loading context: {e}")
            self.initial_context = "Default context: Technology sector engagement"
    
    def _load_slots_data(self):
        """Load slots and definitions from CSV."""
        try:
            slots_url = self._build_url("Test_Slots.csv")
            print(f"Loading slots from: {slots_url}")
            
            response = requests.get(slots_url, timeout=30)
            response.raise_for_status()
            
            slots_csv = StringIO(response.text)
            reader = csv.DictReader(slots_csv)
            
            self.slots = []
            self.definitions = {}
            
            for row in reader:
                field = row["field_name"]
                definition = row["definition"]
                self.slots.append(field)
                self.definitions[field] = definition
            
            print(f"Loaded {len(self.slots)} slots: {', '.join(self.slots)}")
            
        except Exception as e:
            print(f"Error loading slots: {e}")
            # Fallback slots
            self.slots = ["Initiative", "Type of AI", "Current Stage", "Budget", 
                         "Business Objectives", "Success Metrics", "Department", 
                         "Tech Stack", "Next Steps"]
            self.definitions = {slot: f"Information about {slot}" for slot in self.slots}
    
    def _build_url(self, filename: str) -> str:
        """Build Azure blob URL with optional SAS token."""
        if self.sas_token:
            return f"{self.base_url}/{filename}?{self.sas_token}"
        return f"{self.base_url}/{filename}"
    
    def _format_context(self, context_row: Dict[str, str]) -> str:
        """Format context row into readable string."""
        lines = [f"{col}: {val}" for col, val in context_row.items() if val]
        return "Customer context:\n" + "\n".join(lines)
    
    def _setup_prompts(self):
        """Setup LangChain prompts with proper structure."""
        
        # System message for conversation management
        system_message = """You are a professional AI consultant representing the AI Centre of Excellence. 
Your role is to conduct a structured interview to gather comprehensive information about AI initiatives.

CONVERSATION GUIDELINES:
1. Be conversational, professional, and engaging
2. Ask ONE question at a time to collect specific information
3. Build upon previous responses to show active listening
4. Reference the customer's context and previous initiatives when relevant
5. Keep questions focused and avoid overwhelming the customer

RESPONSE FORMAT:
Always respond with just your question or comment - no special formatting, thoughts, or reasoning sections needed.
Keep responses concise and natural.

FIELD COLLECTION ORDER: {field_order}

Your goal is to collect information for each field systematically while maintaining a natural conversation flow."""

        # Create chat prompt template
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Add field order as partial variable
        self.chat_prompt = self.chat_prompt.partial(
            field_order=", ".join(self.slots)
        )
        
        # Extraction prompt for field extraction
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract specific information from the user's response ONLY. 
Do not extract information from the assistant's questions.

Fields to extract: {field_definitions}

Return a JSON object with field names as keys. Use null for fields not mentioned by the user.
Only extract information explicitly stated by the user.

Example format:
{{"Initiative": "chatbot project", "Type of AI": "natural language processing", "Department": null}}"""),
            ("human", "User response: {user_response}\n\nExtract information as JSON:")
        ])
    
    def get_initial_greeting(self) -> str:
        """Generate personalized initial greeting."""
        try:
            # Context for greeting
            context_info = self._build_greeting_context()
            
            # Generate greeting
            greeting_prompt = ChatPromptTemplate.from_messages([
                ("system", """Generate a warm, professional greeting for an AI initiative interview.
Reference the customer context and any previous initiatives if available.
Ask about their current AI initiatives in a conversational way.
Keep it concise and engaging - just 2-3 sentences."""),
                ("human", f"Customer context: {context_info}\n\nGenerate greeting:")
            ])
            
            response = self.llm.invoke(greeting_prompt.format_messages())
            return response.content.strip()
            
        except Exception as e:
            print(f"Error generating greeting: {e}")
            return "Hello! I'd love to learn about the AI initiatives you're currently working on. Could you tell me about a specific AI project you're involved with?"
    
    def _build_greeting_context(self) -> str:
        """Build context for greeting generation."""
        context_parts = [self.initial_context]
        
        if self.previous_initiatives:
            prev_summary = self._summarize_previous_initiatives()
            context_parts.append(f"Previous initiatives: {prev_summary}")
        
        return "\n\n".join(context_parts)
    
    def _summarize_previous_initiatives(self) -> str:
        """Create a brief summary of previous initiatives."""
        if not self.previous_initiatives:
            return "None"
        
        # Get key info from previous initiatives
        initiatives = []
        for init in self.previous_initiatives[:3]:  # Limit to recent 3
            name = init.get("Initiative", "Unknown")
            ai_type = init.get("Type of AI", "")
            stage = init.get("Current Stage", "")
            initiatives.append(f"{name} ({ai_type}, {stage})".strip(" ,()"))
        
        return "; ".join(initiatives)
    
    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and return response with extracted data."""
        try:
            # Handle exit conditions
            if self._is_exit_command(user_input):
                if self.in_follow_up_mode:
                    # In follow-up mode, exit commands should actually complete the session
                    return self._create_final_completion_response()
                else:
                    return self._create_completion_response()
            
            # Add user message to history
            self._add_to_history(HumanMessage(content=user_input))
            
            # If we're in follow-up mode, handle general conversation
            if self.in_follow_up_mode:
                return self._handle_follow_up_conversation(user_input)
            
            # Extract information from user input (only when collecting initial data)
            self._extract_fields_from_input(user_input)
            
            # Check if all fields are collected
            missing_fields = self.get_missing_fields()
            if not missing_fields:
                # Transition to follow-up mode instead of completing
                self.in_follow_up_mode = True
                return self._create_completion_response()
            
            # Generate next question
            next_question = self._generate_next_question(user_input, missing_fields)
            
            # Add assistant response to history
            self._add_to_history(AIMessage(content=next_question))
            
            return {
                "message": next_question,
                "status": "collecting",
                "collected_data": self.collected_data.copy(),
                "missing_fields": missing_fields
            }
            
        except Exception as e:
            print(f"Error processing input: {e}")
            return {
                "message": "I apologize, could you please repeat that?",
                "status": "collecting" if not self.in_follow_up_mode else "follow_up",
                "collected_data": self.collected_data.copy(),
                "missing_fields": self.get_missing_fields()
            }
    
    def _is_exit_command(self, user_input: str) -> bool:
        """Check if user wants to exit the conversation."""
        exit_phrases = ["bye", "exit", "quit", "end", "stop", "done", "finish"]
        input_lower = user_input.lower().strip()
        
        # Don't exit on simple "no" responses - they might be answering a question
        if input_lower in ["no", "n"]:
            # Only exit if no data collected yet or explicitly saying goodbye
            return len([v for v in self.collected_data.values() if v]) == 0
        
        return any(phrase in input_lower for phrase in exit_phrases)
    
    def _extract_fields_from_input(self, user_input: str):
        """Extract field information from user input using LLM."""
        try:
            # Prepare field definitions
            field_defs = "\n".join([f"- {field}: {self.definitions[field]}" 
                                   for field in self.slots])
            
            # Generate extraction prompt
            extraction_messages = self.extraction_prompt.format_messages(
                field_definitions=field_defs,
                user_response=user_input
            )
            
            # Get extraction response with timing
            if ENABLE_LATENCY_LOGGING:
                llm_start = datetime.now()
                response = self.llm.invoke(extraction_messages)
                llm_end = datetime.now()
                llm_latency_ms = (llm_end - llm_start).total_seconds() * 1000
                
                content = response.content.strip()
                
                # Log LLM latency for extraction
                safe_log_event(getattr(self, 'session_id', 'unknown'), "llm_extraction_latency", {
                    "latency_ms": round(llm_latency_ms, 2),
                    "response_length": len(content)
                })
            else:
                response = self.llm.invoke(extraction_messages)
                content = response.content.strip()
            
            # Parse JSON response
            try:
                # Extract JSON from response
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = content[start:end]
                    extracted = json.loads(json_str)
                    
                    # Update collected data
                    for field, value in extracted.items():
                        if value and field in self.slots and not self.collected_data.get(field):
                            cleaned_value = self._clean_field_value(field, value)
                            if cleaned_value:
                                self.collected_data[field] = cleaned_value
                                print(f"Extracted {field}: {cleaned_value}")
                else:
                    # Log when no JSON found in LLM response
                    if ENABLE_EXTRACTION_FAILURE_LOGGING:
                        safe_log_event(getattr(self, 'session_id', 'unknown'), "extraction_failure", {
                            "reason": "no_json_in_response",
                            "llm_response": content[:200]
                        })
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                # Log extraction failure for audit
                if ENABLE_EXTRACTION_FAILURE_LOGGING:
                    safe_log_event(getattr(self, 'session_id', 'unknown'), "extraction_failure", {
                        "reason": "json_parse_error", 
                        "llm_response": content[:200],
                        "error": str(e)
                    })
                
        except Exception as e:
            print(f"Extraction error: {e}")
            # Log extraction failure for audit
            if ENABLE_EXTRACTION_FAILURE_LOGGING:
                safe_log_event(getattr(self, 'session_id', 'unknown'), "extraction_failure", {
                    "reason": "llm_error",
                    "error": str(e)
                })
    
    def _clean_field_value(self, field: str, value: Any) -> Any:
        """Clean and validate field values."""
        if not value or str(value).lower() in ["null", "none", ""]:
            return None
        
        value_str = str(value).strip()
        
        # Field-specific cleaning
        if field == "Budget":
            return self._clean_budget(value_str)
        elif field == "Department":
            return value_str if value_str else "Information Technology"
        elif field in ["Tech Stack", "Success Metrics"]:
            if isinstance(value, list):
                return "; ".join(str(v).strip() for v in value if v)
            return value_str
        
        return value_str
    
    def _clean_budget(self, value: str) -> str:
        """Clean and standardize budget format."""
        if not value:
            return None
        
        # Extract numbers and units
        match = re.search(r'(\d+(?:\.\d+)?)\s*(k|m|b|thousand|million|billion)?', 
                         value.lower())
        if match:
            number, unit = match.groups()
            number = float(number)
            
            # Standardize to millions
            if unit and unit.startswith('k'):
                return f"${number/1000:.1f}M"
            elif unit and unit.startswith('b'):
                return f"${number*1000:.0f}M"
            elif unit and unit.startswith('m'):
                return f"${number:.1f}M"
            else:
                # Assume it's in thousands if no unit
                return f"${number/1000:.1f}M"
        
        return value
    
    def _generate_next_question(self, user_input: str, missing_fields: List[str]) -> str:
        """Generate next question based on conversation context."""
        try:
            # Get next field to focus on
            next_field = missing_fields[0] if missing_fields else None
            if not next_field:
                return "Thank you! I believe I have all the information I need."
            
            # Build context for question generation
            context = self._build_question_context(next_field)
            
            # Create conversation history for prompt (limited length)
            limited_history = self._get_limited_history()
            
            # Generate question using chat prompt
            messages = self.chat_prompt.format_messages(
                history=limited_history,
                input=f"Context: {context}\n\nNext field needed: {next_field} - {self.definitions[next_field]}\n\nUser just said: {user_input}"
            )
            
            # Generate question with timing
            if ENABLE_LATENCY_LOGGING:
                llm_start = datetime.now()
                response = self.llm.invoke(messages)
                llm_end = datetime.now()
                llm_latency_ms = (llm_end - llm_start).total_seconds() * 1000
                
                # Log LLM latency for question generation
                safe_log_event(getattr(self, 'session_id', 'unknown'), "llm_question_latency", {
                    "latency_ms": round(llm_latency_ms, 2),
                    "field": next_field
                })
            else:
                response = self.llm.invoke(messages)
            
            return response.content.strip()
            
        except Exception as e:
            print(f"Error generating question: {e}")
            # Fallback question
            field_def = self.definitions.get(next_field, "this information")
            return f"Could you tell me more about {field_def.lower()}?"
    
    def _build_question_context(self, next_field: str) -> str:
        """Build context for question generation."""
        context_parts = []
        
        # Add customer context
        context_parts.append(self.initial_context)
        
        # Add current progress
        collected = [f"{k}: {v}" for k, v in self.collected_data.items() if v]
        if collected:
            context_parts.append("Information collected so far:\n" + "\n".join(collected))
        
        # Add relevant previous initiatives
        if self.previous_initiatives:
            relevant_prev = self._get_relevant_previous_data(next_field)
            if relevant_prev:
                context_parts.append(f"Previous {next_field} examples: {relevant_prev}")
        
        return "\n\n".join(context_parts)
    
    def _get_relevant_previous_data(self, field: str) -> str:
        """Get relevant data from previous initiatives for the current field."""
        values = []
        for init in self.previous_initiatives[:5]:  # Check recent 5
            if field in init and init[field]:
                values.append(str(init[field]))
        
        return "; ".join(set(values[:3]))  # Unique values, max 3
    
    def _get_limited_history(self) -> List:
        """Get conversation history limited to prevent token overflow."""
        # Keep only recent messages to manage context window
        if len(self.conversation_history) <= self.max_conversation_length:
            return self.conversation_history
        
        # Keep first few and last many messages
        start_messages = self.conversation_history[:5]
        recent_messages = self.conversation_history[-(self.max_conversation_length-5):]
        
        return start_messages + recent_messages
    
    def _add_to_history(self, message):
        """Add message to conversation history."""
        self.conversation_history.append(message)
        
        # Trim history if it gets too long
        if len(self.conversation_history) > self.max_conversation_length + 10:
            self.conversation_history = self.conversation_history[-self.max_conversation_length:]
    
    def _create_completion_response(self) -> Dict[str, Any]:
        """Create response for completed conversation with follow-up question."""
        follow_up_message = "Thank you for connecting with me! I'm here to learn about AI initiatives at your organization. Is there anything about your AI projects or plans you'd like to discuss or share?"
        return {
            "message": follow_up_message,
            "status": "follow_up",  # Changed from "completed" to allow continued conversation
            "collected_data": self.collected_data.copy(),
            "missing_fields": []
        }
    
    def _create_final_completion_response(self) -> Dict[str, Any]:
        """Create final completion response when user wants to end follow-up conversation."""
        return {
            "message": "Thank you for sharing your insights about AI initiatives. Have a great day!",
            "status": "completed",
            "collected_data": self.collected_data.copy(),
            "missing_fields": []
        }
    
    def _handle_follow_up_conversation(self, user_input: str) -> Dict[str, Any]:
        """Handle general conversation after all required fields are collected."""
        try:
            # If we're in the middle of collecting additional initiative data
            if self.collecting_additional:
                return self._handle_additional_initiative_collection(user_input)
            
            # Check if user mentioned a new initiative
            new_initiative = self._detect_new_initiative(user_input)
            if new_initiative:
                return self._offer_to_collect_additional_initiative(new_initiative, user_input)
            
            # Generate a conversational response
            follow_up_response = self._generate_follow_up_response(user_input)
            
            # Add assistant response to history
            self._add_to_history(AIMessage(content=follow_up_response))
            
            return {
                "message": follow_up_response,
                "status": "follow_up",
                "collected_data": self.collected_data.copy(),
                "missing_fields": []
            }
            
        except Exception as e:
            print(f"Error in follow-up conversation: {e}")
            return {
                "message": "That's interesting! Is there anything else you'd like to share about your AI initiatives?",
                "status": "follow_up",
                "collected_data": self.collected_data.copy(),
                "missing_fields": []
            }
    
    def _generate_follow_up_response(self, user_input: str) -> str:
        """Generate conversational response for follow-up discussions."""
        try:
            # Create a prompt for follow-up conversation
            follow_up_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a professional AI consultant having a follow-up conversation after collecting 
all required information about an AI initiative. Be conversational, encouraging, and show interest in their work.

You can:
- Ask follow-up questions about their AI projects
- Provide encouragement or insights
- Discuss challenges or opportunities
- Ask about other AI initiatives they might have

Keep responses concise (1-2 sentences) and engaging. End with a question to keep the conversation going, 
unless they seem to be wrapping up the discussion."""),
                MessagesPlaceholder(variable_name="history"),
                ("human", "User said: {input}")
            ])
            
            # Generate response using limited history
            limited_history = self._get_limited_history()
            messages = follow_up_prompt.format_messages(
                history=limited_history,
                input=user_input
            )
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            print(f"Error generating follow-up response: {e}")
            return "That's very interesting! What other AI projects are you excited about working on?"
    
    def _detect_new_initiative(self, user_input: str) -> str:
        """Detect if user mentions a new AI initiative in their input."""
        try:
            # Use LLM to detect new initiatives
            detection_prompt = ChatPromptTemplate.from_messages([
                ("system", """Analyze the user's message to detect if they mention any NEW AI initiatives or projects 
that are different from what was already discussed.

Look for phrases like:
- "we're also working on..."
- "another project is..."
- "we have a [project type] project"
- "planning to build..."
- "developing a..."

Only extract clear, distinct AI initiatives. Don't extract general AI concepts or technologies.

If you find a new initiative, respond with just the initiative name/description (e.g. "voice assistant", "document analysis system", "fraud detection AI").
If no new initiative is mentioned, respond with "NONE".

Examples:
User: "We're also planning a voice assistant project" → "voice assistant project"
User: "Another initiative is document analysis" → "document analysis initiative" 
User: "That uses machine learning" → "NONE"
User: "We also do chatbots" → "NONE" (if chatbot was already discussed)"""),
                ("human", f"User said: {user_input}")
            ])
            
            response = self.llm.invoke(detection_prompt.format_messages())
            result = response.content.strip()
            
            # Return the initiative if found, None if not
            if result.upper() == "NONE" or not result:
                return None
            
            print(f"Detected new initiative: {result}")
            return result
            
        except Exception as e:
            print(f"Error detecting new initiative: {e}")
            return None
    
    def _offer_to_collect_additional_initiative(self, initiative_name: str, user_input: str) -> Dict[str, Any]:
        """Offer to collect detailed information about the additional initiative."""
        try:
            offer_message = f"That sounds like an exciting project! Would you like me to collect some detailed information about your {initiative_name} as well? I can gather the same type of information we discussed for your previous initiative."
            
            # Start collecting additional initiative
            self.collecting_additional = True
            self.current_additional_data = {field: None for field in self.slots}
            self.current_additional_data["Initiative"] = initiative_name
            
            # Add to conversation history
            self._add_to_history(AIMessage(content=offer_message))
            
            return {
                "message": offer_message,
                "status": "follow_up",
                "collected_data": self.collected_data.copy(),
                "missing_fields": []
            }
            
        except Exception as e:
            print(f"Error offering to collect additional initiative: {e}")
            return {
                "message": f"That {initiative_name} sounds interesting! Tell me more about it.",
                "status": "follow_up", 
                "collected_data": self.collected_data.copy(),
                "missing_fields": []
            }
    
    def _handle_additional_initiative_collection(self, user_input: str) -> Dict[str, Any]:
        """Handle collection using existing code by temporarily switching context."""
        try:
            # Check if user declines to provide details
            if self._is_decline_response(user_input):
                self.collecting_additional = False
                self.current_additional_data = {}
                
                decline_response = "No problem! Is there anything else about your AI initiatives you'd like to discuss?"
                self._add_to_history(AIMessage(content=decline_response))
                
                return {
                    "message": decline_response,
                    "status": "follow_up",
                    "collected_data": self.collected_data.copy(),
                    "missing_fields": []
                }
            
            # Save original state
            original_collected_data = self.collected_data.copy()
            original_in_follow_up = self.in_follow_up_mode
            
            try:
                # Switch to additional initiative collection mode - reuse ALL existing code
                self.collected_data = self.current_additional_data.copy()
                self.in_follow_up_mode = False  # Act like normal collection
                
                # Use existing process_user_input method completely
                temp_result = self.process_user_input(user_input)
                
                # Copy collected data back
                self.current_additional_data = self.collected_data.copy()
                
                # If collection completed, save and continue follow-up
                if temp_result['status'] == 'follow_up':  # All fields collected
                    return self._complete_additional_initiative_collection()
                
                # Return the question but keep in follow_up status
                return {
                    "message": temp_result['message'],
                    "status": "follow_up",
                    "collected_data": original_collected_data,
                    "missing_fields": []
                }
                
            finally:
                # Always restore original state
                self.collected_data = original_collected_data
                self.in_follow_up_mode = original_in_follow_up
            
        except Exception as e:
            print(f"Error handling additional initiative collection: {e}")
            return {
                "message": "Could you please repeat that?",
                "status": "follow_up",
                "collected_data": self.collected_data.copy(),
                "missing_fields": []
            }
    
    def _is_decline_response(self, user_input: str) -> bool:
        """Check if user declines to provide additional details."""
        decline_phrases = ["no", "not now", "skip", "maybe later", "not interested", "no thanks"]
        return any(phrase in user_input.lower() for phrase in decline_phrases)
    
    def _is_agreement_response(self, user_input: str) -> bool:
        """Check if user agrees to provide additional details."""
        agree_phrases = ["yes", "sure", "ok", "okay", "sounds good", "let's do it", "go ahead"]
        return any(phrase in user_input.lower() for phrase in agree_phrases)
    
    
    def _complete_additional_initiative_collection(self) -> Dict[str, Any]:
        """Complete collection of additional initiative and save to CSV."""
        try:
            # Save the additional initiative to CSV
            print(f"Saving additional initiative: {self.current_additional_data}")
            
            def save_additional_csv():
                try:
                    print(f"Starting background CSV write for additional initiative")
                    self.update_initiatives_csv(self.current_additional_data)
                    print(f"Additional initiative saved to CSV successfully")
                except Exception as e:
                    print(f"Failed to save additional initiative to CSV: {e}")
            
            # Save in background
            csv_thread = threading.Thread(target=save_additional_csv)
            csv_thread.daemon = True
            csv_thread.start()
            
            # Add to additional initiatives list
            self.additional_initiatives.append(self.current_additional_data.copy())
            
            # Reset collection state
            initiative_name = self.current_additional_data.get("Initiative", "the initiative")
            self.collecting_additional = False
            self.current_additional_data = {}
            
            completion_message = f"Thank you! I've captured all the details about the {initiative_name}. Is there anything else about your AI projects you'd like to discuss?"
            
            self._add_to_history(AIMessage(content=completion_message))
            
            return {
                "message": completion_message,
                "status": "follow_up",
                "collected_data": self.collected_data.copy(),
                "missing_fields": []
            }
            
        except Exception as e:
            print(f"Error completing additional initiative collection: {e}")
            return {
                "message": "Thank you for sharing that information! Is there anything else you'd like to discuss?",
                "status": "follow_up",
                "collected_data": self.collected_data.copy(),
                "missing_fields": []
            }
    
    def get_missing_fields(self) -> List[str]:
        """Get list of fields that still need to be collected."""
        missing = []
        for field, value in self.collected_data.items():
            if not value or (isinstance(value, str) and not value.strip()):
                missing.append(field)
        return missing
    
    def format_current_status(self) -> str:
        """Format current collection status."""
        collected = []
        for field, value in self.collected_data.items():
            if value:
                collected.append(f"{field}: {value}")
        
        return "\n".join(collected) if collected else "No information collected yet."
    
    def _load_previous_initiatives(self) -> List[Dict[str, Any]]:
        """Load previous initiatives from storage."""
        try:
            initiatives_url = self._build_url("AI_Initiatives.csv")
            response = requests.get(initiatives_url, timeout=30)
            
            if response.status_code == 200:
                initiatives_csv = StringIO(response.text)
                df = pd.read_csv(initiatives_csv)
                print(f"Loaded {len(df)} previous initiatives")
                return df.to_dict('records')
            else:
                print("No previous initiatives found")
                return []
                
        except Exception as e:
            print(f"Error loading previous initiatives: {e}")
            return []
    
    def update_initiatives_csv(self, initiative_data: Dict[str, Any]):
        """Update AI_Initiatives.csv with new initiative in Azure Blob Storage."""
        try:
            import pandas as pd
            from io import StringIO
            import requests
            
            # Azure Blob Storage URLs with SAS token
            sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN", "")
            base_url = "https://acu1tmagentbotd1saglobal.blob.core.windows.net/csvdata"
            
            if not sas_token:
                print("ERROR: No SAS token available - cannot write to Azure Blob")
                return False
                
            initiatives_csv_url = f"{base_url}/AI_Initiatives.csv?{sas_token}"
            
            print(f"DEBUG: Updating CSV at Azure Blob Storage")
            
            # Read existing initiatives from Azure Blob
            try:
                initiatives_response = requests.get(initiatives_csv_url, timeout=30)
                if initiatives_response.status_code == 200:
                    initiatives_csv = StringIO(initiatives_response.text)
                    df = pd.read_csv(initiatives_csv)
                    print(f"DEBUG: Read {len(df)} existing initiatives from Azure Blob")
                else:
                    # File doesn't exist yet, create new DataFrame
                    df = pd.DataFrame(columns=self.slots)
                    print(f"DEBUG: No existing initiatives file found, creating new one")
            except Exception as e:
                print(f"DEBUG: Error reading from Azure Blob, creating new DataFrame: {e}")
                df = pd.DataFrame(columns=self.slots)
            
            # Check if initiative already exists
            if 'Initiative' in initiative_data and initiative_data['Initiative']:
                initiative_name = initiative_data['Initiative']
                existing_idx = df[df['Initiative'] == initiative_name].index
                
                if len(existing_idx) > 0:
                    # Update existing initiative
                    print(f"DEBUG: Updating existing initiative: {initiative_name}")
                    for field, value in initiative_data.items():
                        if value is not None and value != "":
                            df.loc[existing_idx[0], field] = value
                else:
                    # Add new initiative
                    print(f"DEBUG: Adding new initiative: {initiative_name}")
                    new_row = {field: None for field in self.slots}
                    new_row.update(initiative_data)
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                # Add as new initiative without name
                print(f"DEBUG: Adding new initiative without name")
                new_row = {field: None for field in self.slots}
                new_row.update(initiative_data)
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save updated DataFrame back to Azure Blob
            csv_string = df.to_csv(index=False)
            print(f"DEBUG: Generated CSV data with {len(df)} initiatives")
            
            # Upload back to Azure Blob using PUT request
            if sas_token:
                upload_url = f"{base_url}/AI_Initiatives.csv?{sas_token}"
                headers = {
                    'x-ms-blob-type': 'BlockBlob',
                    'Content-Type': 'text/csv'
                }
                
                upload_response = requests.put(upload_url, data=csv_string, headers=headers, timeout=30)
                
                if upload_response.status_code in [200, 201]:
                    print(f"DEBUG: Successfully uploaded AI_Initiatives.csv to Azure Blob")
                    return True
                else:
                    print(f"DEBUG: Failed to upload to Azure Blob. Status: {upload_response.status_code}")
                    print(f"DEBUG: Response: {upload_response.text}")
                    return False
            else:
                print(f"DEBUG: No SAS token available - cannot upload to Azure Blob")
                return False
                
        except Exception as e:
            print(f"ERROR: Error updating CSV file: {str(e)}")
            import traceback
            print(f"DEBUG: Full traceback: {traceback.format_exc()}")
            return False


class ConversationSession:
    """Manages individual conversation sessions with proper state management."""
    
    def __init__(self, session_id: str, user_id: str = None):
        self.session_id = session_id
        self.user_id = user_id
        self.agent = SurveyAgent()
        self.agent.session_id = session_id  # Pass session_id to agent for logging
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.status = "active"  # active, follow_up, completed, error
        self.is_first_turn = True
        
        # Start conversation logging
        log_conversation_event(session_id, "session_started", {
            "user_id": user_id,
            "timestamp": self.created_at.isoformat()
        })
        
    def get_initial_greeting(self) -> str:
        """Get initial greeting for the session."""
        try:
            greeting = self.agent.get_initial_greeting()
            self.is_first_turn = False
            self._update_activity()
            
            # Log the initial greeting
            log_bot_question(self.session_id, greeting, {"type": "initial_greeting"})
            
            return greeting
        except Exception as e:
            print(f"Error getting greeting: {e}")
            return "Hello! I'd love to learn about your AI initiatives. Could you tell me about a project you're working on?"
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and return response."""
        start_time = datetime.now() if ENABLE_LATENCY_LOGGING else None
        try:
            self._update_activity()
            
            # Log user input
            log_user_answer(self.session_id, user_input)
            
            # Process through agent
            result = self.agent.process_user_input(user_input)
            
            # Log bot response
            log_bot_question(self.session_id, result["message"], {
                "status": result["status"],
                "missing_fields": result.get("missing_fields", []),
                "collected_fields": len([v for v in result.get("collected_data", {}).values() if v])
            })
            
            # Log any extracted fields with the user's answer
            collected_data = result.get("collected_data", {})
            if collected_data:
                non_empty_fields = {k: v for k, v in collected_data.items() if v}
                if non_empty_fields:
                    # Update the last user answer log with extracted fields
                    log_user_answer(self.session_id, user_input, non_empty_fields, {
                        "extraction_successful": True,
                        "fields_count": len(non_empty_fields)
                    })
            
            # Update session status
            if result["status"] == "follow_up":
                self.status = "follow_up"
                # When entering follow-up mode, save the collected data to CSV
                def csv_with_error_handling():
                    try:
                        print(f"Starting background CSV write for session {self.session_id} (follow-up mode)")
                        self.agent.update_initiatives_csv(result["collected_data"])
                        print(f"Background CSV write completed for session {self.session_id}")
                    except Exception as e:
                        print(f"Background CSV write failed for session {self.session_id}: {e}")
                
                csv_thread = threading.Thread(target=csv_with_error_handling)
                csv_thread.daemon = True
                csv_thread.start()
                
                # Log transition to follow-up mode
                log_conversation_event(self.session_id, "entered_follow_up_mode", {
                    "completion_reason": "all_fields_collected",
                    "collected_data": result["collected_data"]
                })
                
            elif result["status"] == "completed":
                self.status = "completed"
                
                # Log conversation completion and save conversation log
                log_conversation_event(self.session_id, "session_completed", {
                    "completion_reason": "user_ended_conversation",
                    "final_data": result["collected_data"]
                })
                
                # Save the complete conversation log to Azure Blob
                def save_log_with_error_handling():
                    try:
                        print(f"Saving conversation log for session {self.session_id}")
                        save_conversation(self.session_id, self.user_id)
                        print(f"Conversation log saved for session {self.session_id}")
                    except Exception as e:
                        print(f"Failed to save conversation log for session {self.session_id}: {e}")
                
                log_thread = threading.Thread(target=save_log_with_error_handling)
                log_thread.daemon = True
                log_thread.start()
            
            # Log processing latency (thread-safe)
            if ENABLE_LATENCY_LOGGING and start_time:
                end_time = datetime.now()
                latency_ms = (end_time - start_time).total_seconds() * 1000
                safe_log_event(self.session_id, "processing_latency", {
                    "total_ms": round(latency_ms, 2),
                    "status": result["status"],
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat()
                })
            
            return result
            
        except Exception as e:
            print(f"Error processing input: {e}")
            self.status = "error"
            return {
                "message": "I apologize, but I encountered an error. Could you please try again?",
                "status": "error",
                "collected_data": self.agent.collected_data.copy(),
                "missing_fields": self.agent.get_missing_fields()
            }
    
    def _update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
    
    def is_expired(self, timeout_hours: int = 2) -> bool:
        """Check if session has expired."""
        cutoff = datetime.now() - timedelta(hours=timeout_hours)
        return self.last_activity < cutoff
    
    def get_status(self) -> Dict[str, Any]:
        """Get current session status."""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "collected_data": self.agent.collected_data.copy(),
            "missing_fields": self.agent.get_missing_fields(),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat()
        }


# Test function
def test_survey_agent():
    """Test the survey agent implementation."""
    print("Testing SurveyAgent...")
    
    try:
        # Create session
        session = ConversationSession("test-session")
        
        # Test initial greeting
        greeting = session.get_initial_greeting()
        print(f"Initial greeting: {greeting}")
        
        # Test conversation flow
        test_inputs = [
            "We're working on a chatbot project",
            "It uses natural language processing and machine learning",
            "We're in the development phase",
            "The budget is around 500k",
            "We want to improve customer service efficiency",
            "We'll measure response time and satisfaction scores",
            "It's being developed by the IT department",
            "We're using Python, TensorFlow, and Azure",
            "Next we need to do user testing"
        ]
        
        for i, test_input in enumerate(test_inputs):
            print(f"\n--- Turn {i+1} ---")
            print(f"User: {test_input}")
            
            result = session.process_input(test_input)
            print(f"Assistant: {result['message']}")
            print(f"Status: {result['status']}")
            print(f"Missing fields: {result['missing_fields']}")
            
            if result['status'] == 'completed':
                print("Conversation completed!")
                break
        
        # Print final collected data
        print(f"\nFinal collected data: {session.agent.collected_data}")
        
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_survey_agent()