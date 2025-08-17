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
from datetime import datetime, timedelta


class SurveyAgent:
    """Enhanced SurveyAgent with proper LangChain implementation and conversation management."""
    
    def __init__(self):
        # Initialize LLM with better settings
        self.llm = ChatOpenAI(
            temperature=0.1,  # Slight creativity for natural conversation
            model="gpt-4",
            max_tokens=1000,
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
                return self._create_completion_response()
            
            # Add user message to history
            self._add_to_history(HumanMessage(content=user_input))
            
            # Extract information from user input
            self._extract_fields_from_input(user_input)
            
            # Check if all fields are collected
            missing_fields = self.get_missing_fields()
            if not missing_fields:
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
                "status": "collecting",
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
            
            # Get extraction response
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
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                
        except Exception as e:
            print(f"Extraction error: {e}")
    
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
        """Create response for completed conversation."""
        return {
            "message": "Thank you! I've gathered all the information about your AI initiative. This will be very helpful for our analysis.",
            "status": "completed",
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
        """Update initiatives CSV with new data."""
        try:
            # This would implement the CSV update logic
            # For now, just log the data
            print(f"Would update CSV with: {initiative_data}")
            return True
            
        except Exception as e:
            print(f"Error updating CSV: {e}")
            return False


class ConversationSession:
    """Manages individual conversation sessions with proper state management."""
    
    def __init__(self, session_id: str, user_id: str = None):
        self.session_id = session_id
        self.user_id = user_id
        self.agent = SurveyAgent()
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.status = "active"  # active, completed, error
        self.is_first_turn = True
        
    def get_initial_greeting(self) -> str:
        """Get initial greeting for the session."""
        try:
            greeting = self.agent.get_initial_greeting()
            self.is_first_turn = False
            self._update_activity()
            return greeting
        except Exception as e:
            print(f"Error getting greeting: {e}")
            return "Hello! I'd love to learn about your AI initiatives. Could you tell me about a project you're working on?"
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and return response."""
        try:
            self._update_activity()
            
            # Process through agent
            result = self.agent.process_user_input(user_input)
            
            # Update session status
            if result["status"] == "completed":
                self.status = "completed"
                # Update CSV in background (you could make this async)
                self.agent.update_initiatives_csv(result["collected_data"])
            
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