import csv
import json
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
import re
import requests
from io import StringIO

class SurveyAgent:
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(temperature=0)
        
        # Azure Blob Storage URLs with optional SAS token support
        sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN", "")
        base_url = "https://acu1tmagentbotd1saglobal.blob.core.windows.net/csvdata"
        
        # Construct URLs with SAS token if available
        if sas_token:
            context_csv_url = f"{base_url}/Test_Context_HPE.csv?{sas_token}"
            slots_csv_url = f"{base_url}/Test_Slots.csv?{sas_token}"
        else:
            # Use environment variables or default URLs
            context_csv_url = os.getenv("CONTEXT_CSV_URL", f"{base_url}/Test_Context_HPE.csv")
            slots_csv_url = os.getenv("SLOTS_CSV_URL", f"{base_url}/Test_Slots.csv")
        
        print(f"Loading context from: {context_csv_url}")
        print(f"Loading slots from: {slots_csv_url}")
        
        # Read Initial context from CSV via HTTP
        context_response = requests.get(context_csv_url, timeout=30)
        context_response.raise_for_status()
        print(f"Successfully loaded context CSV, size: {len(context_response.text)} chars")
        
        context_csv = StringIO(context_response.text)
        reader = csv.DictReader(context_csv)
        context_row = next(reader)

        self.initial_context_lines = [f"{col}: {val}" for col, val in context_row.items()]
        self.initial_context = "Initial context:\n" + "\n".join(self.initial_context_lines)

        # Read slots and definitions from CSV via HTTP
        self.slots = []
        self.definitions = {}
        slots_response = requests.get(slots_csv_url, timeout=30)
        slots_response.raise_for_status()
        print(f"Successfully loaded slots CSV, size: {len(slots_response.text)} chars")
        
        slots_csv = StringIO(slots_response.text)
        reader = csv.DictReader(slots_csv)
        for row in reader:
            field = row["field_name"]
            desc = row["definition"]
            self.slots.append(field)
            self.definitions[field] = desc
            
        print(f"Loaded {len(self.slots)} slots: {', '.join(self.slots)}")

        # Initialize collected data
        self.collected_data = {field: None for field in self.slots}

        # Load previous initiatives
        self.previous_initiatives = self.load_previous_initiatives()
        self.previous_initiatives_context = self.format_previous_initiatives_context(self.previous_initiatives)
        
        # Update system message to enforce better context usage
        self.system_message = """You are a friendly, conversational assistant representing the AI Centre of Excellence at our Software Services company. You speak to Customer Account Reps to gather AI-initiative details. At each turn, think step by step and emit your reasoning before asking your question.

        Previous Initiatives Context:
        {previous_initiatives_context}

        Current context:  
        {initial_context}

            1. FIRST TURN ONLY:
        - **MUST** start with a personalized greeting using the initial context:
            * Address the Customer Account Rep directly
            * Reference their account (company name) and industry from initial context
            * Reference specific business details from initial context
            * **MUST** reference specific details from previous initiatives ONLY if they exist in the CSV file
            * **MUST** ask about new initiatives in context of previous ones ONLY if they exist
            - **Format**:
                ```
            Thought 1: <analyze initial context to identify:
                - Account name and industry from initial context
                - Key business details from initial context
                - Previous initiatives from previous_initiatives_context (if any)
                - How to connect these to new initiatives>
            Thought 2: <analyze previous initiatives from previous_initiatives_context to identify:
                - Types of AI used in previous initiatives
                - Success patterns and challenges
                - Budget ranges and stages
                - Tech stack patterns
                - How to leverage this knowledge for new initiatives>
            Answer: <greeting addressing the rep> <reference specific previous initiatives ONLY if they exist in CSV> <ask about new initiatives in context>
            ```
        - **Example**:
            ```
            Thought 1: From the initial context, I see this is a conversation with a Customer Account Rep for HPE in the Technology sector. They have previous initiatives in [specific details from previous_initiatives_context]. I should acknowledge their previous work and ask how their new initiative relates to or builds upon these.
            
            Thought 2: Looking at their previous initiatives, I notice they've worked with [specific AI types], achieved [specific successes], and faced [specific challenges]. Their typical budget range is [range], and they commonly use [tech stack]. I should use this to guide my questions about their new initiative.
            
            Answer: Hello! I see you're working with HPE in the Technology sector. I notice you've previously worked on [specific previous initiative details]. How does your new AI initiative build upon or differ from these previous projects?
                ```

            2. ALL SUBSEQUENT TURNS:
        - **MUST** focus on collecting ONE field at a time
        - **MUST** follow the field order from the CSV file: {field_order}
        - **MUST** maintain context throughout the conversation
            - **Chain of Thought + Question**  
            1. Thought 1: Analyze the user's last response:
                - What specific information did they provide?
                - How does it connect to their business context?
                - How does it compare to their previous initiatives?
                - What patterns or trends are emerging?
            2. Thought 2: Analyze previous initiatives for context:
                - What similar fields were used in previous initiatives?
                - What were the typical values or ranges?
                - What patterns or trends can we learn from?
                - How can we use this to guide our questions?
            3. Thought 3: Identify the next missing field and its importance:
                - Why is this field relevant to what they've shared?
                - How does it connect to their business objectives?
                - How does it relate to their previous initiatives?
                - What specific aspects should we focus on?
            4. Thought 4: Generate insights by connecting:
                - Their business context from initial context
                - Previous initiatives and their outcomes
                - Their current initiative's details
                - Industry best practices
                - Their specific business needs
                - Patterns in their AI adoption
                - Success factors from previous initiatives
            5. Thought 5: Craft a contextual question that:
                - Focuses ONLY on the next missing field
                - Builds on their previous responses
                - Shows understanding of their business context
                - Connects to their specific situation
                - Uses insights from previous turns
                - References relevant previous initiatives
                - Makes the question relevant to their goals
                - Leverages learnings from past initiatives
                - **MUST** incorporate insights from Thoughts 1-4
                - **MUST** show understanding of their specific situation
                - **MUST** connect to their previous responses
                - **MUST** use patterns from previous initiatives if available
                - **MUST** compare with previous initiative's values for the same field
                - **MUST** ask follow-up questions when patterns are detected
                - **MUST** maintain business context throughout

            - **Format**:
                ```
            Thought 1: <analysis of their last response, business context, and previous initiatives>
            Thought 2: <analysis of previous initiatives for context and patterns>
            Thought 3: <importance of next field in their context and previous initiatives>
            Thought 4: <insights connecting their business, previous initiatives, and current situation>
            Thought 5: <how to frame the question using these insights and previous context>
            Answer: <natural, contextual question that shows understanding and builds on previous responses>
                ```

            - When constructing your question:
            * Make it specific to their business context
            * Connect it to their previous responses
            * Reference relevant previous initiatives
            * Show understanding of their industry and needs
            * Use insights from previous turns
            * Make it feel like a natural conversation
            * Avoid generic or formulaic questions
            * Don't repeat their company name unnecessarily
            * Focus on the business value and impact
            * Leverage patterns from previous initiatives
            * Compare with previous initiative's values
            * Ask follow-up questions when patterns are detected
            * Maintain business context throughout

            3. EXIT CONDITIONS:
            - If the user says "no", "bye", "exit", or "quit":
            1. Thought 1: Review what we've learned about their initiative
            2. Thought 2: Check for any critical missing information
            3. Thought 3: Consider if we need to ask about specific aspects
            4. Answer: <natural exit or continuation prompt that acknowledges their situation>

        

            Current status:  
            {current_status}

            Missing fields:  
            {missing_fields}

            Conversation history:  
            {history}
            """
        
        # Create chat prompt template
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        # Add field_order as a partial variable
        self.chat_prompt = self.chat_prompt.partial(
            field_order=", ".join(self.slots)
        )

        # Extraction prompt for pulling all mentioned fields into JSON
        self.extraction_template = """
            Extract information about the following fields from this conversation snippet. 
            Return a JSON object with the following rules:
            1. Use EXACT field names as keys
            2. For each field, extract the most recent/relevant information
            3. If a field is not mentioned, return null for that field
            4. For fields that can have multiple values, return them as a JSON array
            5. Preserve the exact values mentioned in the conversation
            6. DO NOT include default values - only extract values explicitly mentioned

            Fields with definitions:
            {fields_with_definitions}

            Conversation snippet:
            {snippet}

            Example output format:
            {{
                "Initiative": "project name",
                "Type of AI": null,
                "Current Stage": null,
                "Budget": null,
                "Business Objectives": null,
                "Success Metrics": null,
                "Department": null,
                "Tech Stack": null,
                "Next Steps": null
            }}

            Return ONLY valid JSON with keys matching the field names exactly.
            """

        self.extraction_prompt = PromptTemplate(
            input_variables=["fields_with_definitions", "snippet"],
            template=self.extraction_template
        )

        # Add field-specific cleaning functions
        self.field_cleaners = {
            'Budget': self._clean_budget,
            'Success Metrics': self._clean_success_metrics,
            'Tech Stack': self._clean_tech_stack,
            'Department': self._clean_department
        }

        # Add transcript storage
        self.transcript = []

    def _clean_budget(self, value):
        """Clean and standardize budget format."""
        if not value:
            return None
        
        # Handle incremental changes
        if 'increase' in str(value).lower() or 'additional' in str(value).lower():
            # Extract the increment amount
            match = re.search(r'(\d+(?:\.\d+)?)\s*(k|m|b)?', str(value).lower())
            if match:
                number, unit = match.groups()
                number = float(number)
                # Convert to standard format
                if unit == 'k':
                    return f"${number/1000:.1f}M"
                elif unit == 'm':
                    return f"${number}M"
                elif unit == 'b':
                    return f"${number*1000}M"
                else:
                    return f"${number/1000000:.1f}M"
        
        # Handle regular budget values
        value = str(value).lower()
        match = re.search(r'(\d+(?:\.\d+)?)\s*(k|m|b)?', value)
        if match:
            number, unit = match.groups()
            number = float(number)
            # Convert to standard format
            if unit == 'k':
                return f"${number/1000:.1f}M"
            elif unit == 'm':
                return f"${number}M"
            elif unit == 'b':
                return f"${number*1000}M"
            else:
                return f"${number/1000000:.1f}M"
        return value

    def _clean_success_metrics(self, value):
        """Clean and standardize success metrics."""
        if not value:
            return None
        if isinstance(value, list):
            # Clean each item in the list and join with semicolons
            cleaned_items = [str(v).strip() for v in value if v]
            return "; ".join(cleaned_items)
        return str(value).strip()

    def _clean_tech_stack(self, value):
        """Clean and standardize tech stack entries."""
        if not value:
            return None
        if isinstance(value, list):
            # Clean each item in the list and join with semicolons
            cleaned_items = [str(v).strip().title() for v in value if v]
            return "; ".join(cleaned_items)
        return str(value).strip().title()

    def _clean_department(self, value):
        """Clean and standardize department value."""
        if not value:
            return "Information Technology"  # Default only if no value provided
        return str(value).strip()  # Otherwise use the provided value

    def format_current_status(self):
        lines = []
        for field, value in self.collected_data.items():
            if value not in (None, "", []):
                if isinstance(value, list):
                    val_str = ", ".join(value)
                else:
                    val_str = str(value)
                lines.append(f"{field}: {val_str}")
        return "\n".join(lines) if lines else "No information collected yet."

    def get_missing_fields(self):
        missing = []
        for field, value in self.collected_data.items():
            if value is None or value == "" or (isinstance(value, list) and len(value) == 0):
                missing.append(field)
        return missing

    def validate_field_value(self, field, value):
        """Validate a field value against its definition and previous values."""
        if not value:
            return False
            
        # Get the field definition
        definition = self.definitions.get(field, "").lower()
        
        # Special handling for specific fields
        if field == 'Budget':
            # Look for any number with optional $ and k/m/b
            import re
            return bool(re.search(r'\$?\s*\d+(?:\.\d+)?\s*(?:k|m|b)?', str(value), re.IGNORECASE))
        
        elif field == 'Current Stage':
            stages = ['ideation', 'planning', 'development', 'testing', 'uat', 'production', 'pilot']
            return any(stage in str(value).lower() for stage in stages)
        
        elif field == 'Initiative':
            # Don't accept "no new initiatives" as a valid initiative name
            if 'no new initiatives' in str(value).lower():
                return False
        
        # For all other fields, ensure the response is substantial
        return len(str(value)) > 5

    def extract_all_fields_working_copy(self, snippet):
        """Extract field values from conversation snippet using LLM."""
        # Create the fields with definitions string
        fwd = "\n".join([f"- {f}: {self.definitions[f]}" for f in self.slots])
        
        # Format the extraction prompt
        prompt_text = self.extraction_prompt.format(
            fields_with_definitions=fwd,
            snippet=snippet
        )
        
        print("\nDEBUG: Extraction prompt:")
        print(f"DEBUG: Snippet being analyzed: {snippet}")
        
        # Get response from LLM
        response = self.llm.invoke(prompt_text)
        content = response.content.strip()
        
        try:
            # Extract JSON from response
            start = content.find("{")
            end = content.rfind("}") + 1
            json_str = content[start:end]
            extracted = json.loads(json_str)
            
            print(f"DEBUG: Parsed JSON: {extracted}")
            
            # Process each field
            for field, val in extracted.items():
                # Only process fields that have valid values extracted by the LLM
                if val is not None and val != "":
                    print(f"\nDEBUG: Processing field: {field}")
                    print(f"DEBUG: Raw value: {val}")
                    
                    # Only store if field is actually missing (prevents infinite loops)
                    if field not in self.get_missing_fields():
                        print(f"DEBUG: Field {field} already collected, skipping")
                        continue
                    
                    # Special handling for "no new initiatives" response
                    if field == 'Initiative' and 'no new initiatives' in str(val).lower():
                        continue
                    
                    # Clean the value before storing
                    if field in self.field_cleaners:
                        val = self.field_cleaners[field](val)
                        print(f"DEBUG: Cleaned value: {val}")
                    
                    # Store the value
                    self.collected_data[field] = val
                    print(f"DEBUG: Final stored value: {self.collected_data[field]}")
        
        except Exception as e:
            print(f"DEBUG: Error in extraction: {str(e)}")
            print(f"DEBUG: Error type: {type(e).__name__}")
            import traceback
            print(f"DEBUG: Full traceback: {traceback.format_exc()}")
            # Don't return - continue processing to avoid infinite loops
            return

    def extract_all_fields(self, snippet, current_field=None):
        # Split the snippet into user and assistant messages
        messages = [msg.strip() for msg in snippet.split('\n') if msg.strip()]
        
        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.startswith('User:'):
                user_message = msg.replace('User:', '').strip()
                break
        
        # Get the last assistant message before the user message
        assistant_message = None
        for msg in reversed(messages):
            if msg.startswith('Assistant:') and msg != messages[-1]:
                assistant_message = msg.replace('Assistant:', '').strip()
                break
        
        if not user_message or not assistant_message:
            return
        
        # Only process the current field
        if current_field:
            # Skip empty or invalid responses
            if not user_message or user_message.lower() in ['ok', 'yes', 'y', 'no', 'n', 'bye', 'exit', 'quit']:
                return
            
            # Validate the response against the field definition
            if self.validate_field_value(current_field, user_message):
                # Clean the value before storing
                cleaned_value = user_message
                if current_field in self.field_cleaners:
                    cleaned_value = self.field_cleaners[current_field](user_message)
                
                # Store the value appropriately based on field type
                if current_field in ['Tech Stack', 'Success Metrics']:
                    if isinstance(cleaned_value, list):
                        self.collected_data[current_field] = cleaned_value
                    else:
                        self.collected_data[current_field] = [cleaned_value]
                else:
                    self.collected_data[current_field] = cleaned_value

    def update_initiatives_csv(self, new_initiative):
        """Update AI_Initiatives.csv with new initiative or update existing one."""
        try:
            # Use Azure Blob Storage URL for AI_Initiatives.csv like other CSV files
            sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN", "")
            base_url = "https://acu1tmagentbotd1saglobal.blob.core.windows.net/csvdata"
            
            if sas_token:
                initiatives_csv_url = f"{base_url}/AI_Initiatives.csv?{sas_token}"
            else:
                initiatives_csv_url = os.getenv("INITIATIVES_CSV_URL", f"{base_url}/AI_Initiatives.csv")
            
            # Read existing initiatives from Azure Blob
            try:
                initiatives_response = requests.get(initiatives_csv_url, timeout=30)
                if initiatives_response.status_code == 200:
                    initiatives_csv = StringIO(initiatives_response.text)
                    df = pd.read_csv(initiatives_csv)
                    print(f"\nDEBUG: Read {len(df)} existing initiatives from Azure Blob CSV")
                else:
                    # File doesn't exist yet, create new DataFrame
                    df = pd.DataFrame(columns=self.slots)
                    print(f"\nDEBUG: No existing initiatives file found, creating new one")
            except Exception as e:
                print(f"\nDEBUG: Error reading from Azure Blob, creating new DataFrame: {e}")
                df = pd.DataFrame(columns=self.slots)
            
            # Check if initiative already exists
            if 'Initiative' in new_initiative and new_initiative['Initiative']:
                initiative_name = new_initiative['Initiative']
                existing_idx = df[df['Initiative'] == initiative_name].index
                
                if len(existing_idx) > 0:
                    # Update existing initiative
                    print(f"\nDEBUG: Updating existing initiative: {initiative_name}")
                    for field, value in new_initiative.items():
                        if value is not None and value != "":
                            df.loc[existing_idx[0], field] = value
                    print(f"\nDEBUG: Updated fields: {[field for field, value in new_initiative.items() if value is not None and value != '']}")
            else:
                    # Add new initiative
                    print(f"\nDEBUG: Adding new initiative: {initiative_name}")
                    new_row = {field: None for field in self.slots}
                    new_row.update(new_initiative)
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save updated DataFrame back to Azure Blob
            csv_string = df.to_csv(index=False)
            print(f"\nDEBUG: Generated CSV data with {len(df)} initiatives:")
            print(f"DEBUG: CSV content preview: {csv_string[:200]}...")
            
            # Try to upload back to Azure Blob using PUT request
            try:
                if sas_token:
                    upload_url = f"{base_url}/AI_Initiatives.csv?{sas_token}"
                    headers = {
                        'x-ms-blob-type': 'BlockBlob',
                        'Content-Type': 'text/csv'
                    }
                    
                    upload_response = requests.put(upload_url, data=csv_string, headers=headers, timeout=30)
                    
                    if upload_response.status_code in [200, 201]:
                        print(f"\nDEBUG: Successfully uploaded AI_Initiatives.csv to Azure Blob")
                    else:
                        print(f"\nDEBUG: Failed to upload to Azure Blob. Status: {upload_response.status_code}")
                        print(f"DEBUG: Response: {upload_response.text}")
                else:
                    print(f"\nDEBUG: No SAS token available - cannot upload to Azure Blob")
            except Exception as upload_error:
                print(f"\nDEBUG: Error uploading to Azure Blob: {upload_error}")
                print(f"DEBUG: Data will be lost - consider implementing local backup")
            
            print(f"\nDEBUG: Successfully processed initiative data with {len(df)} records")
            
        except Exception as e:
            print(f"\nDEBUG: Error updating CSV file: {str(e)}")
            import traceback
            print(f"DEBUG: Full traceback: {traceback.format_exc()}")

    def run_survey(self):
        all_initiatives = []
        
        while True:
            # Initial greeting and question
            start = input(
                "Hi! My name is Survey Agent. I'm here to collect information about the AI initiatives across the company. "
                "Do you have a new AI initiative in your account to share? (yes/no): "
            ).strip()
            
            # Add to transcript
            self.transcript.append(f"System: {start}")
            
            if start.lower() in ("no", "n", "bye", "exit", "quit"):
                break

            # Reset collected data for new initiative
            self.collected_data = {field: None for field in self.slots}
            messages = []

            # First turn: explicitly build and send the prompt with empty history
            current_status = self.format_current_status()
            missing = self.get_missing_fields()
            next_field = missing[0] if missing else None
            
            print(f"\nDEBUG: Current status: {current_status}")
            print(f"DEBUG: Missing fields: {missing}")
            print(f"DEBUG: Next field to collect: {next_field}")
            
            # Create the first message with empty history and just the initial greeting
            first_prompt = self.chat_prompt.format_messages(
                initial_context=self.initial_context,
                current_status=current_status,
                missing_fields=", ".join(missing),
                history=[],
                input="",  # Empty input for first turn
                previous_initiatives_context=self.previous_initiatives_context
            )
            
            response = self.llm.invoke(first_prompt)
            assistant_msg = response.content.strip()
            
            # Print thoughts separately
            print("\nDEBUG: Assistant's thoughts:")
            thoughts = assistant_msg.split("Thought")
            for thought in thoughts[1:]:  # Skip the first split which is empty
                print(f"DEBUG: Thought{thought.strip()}")
            
            # Print the final answer
            answer = thoughts[-1].split("Answer:")[-1].strip()
            print(f"\nAssistant: {answer}")
            self.transcript.append(f"Assistant: {assistant_msg}")
            
            messages.extend(first_prompt)
            messages.append(AIMessage(content=assistant_msg))

            # Conversation loop
            while True:
                user_input = input("User: ").strip()
                self.transcript.append(f"User: {user_input}")
                
                if user_input.lower() in ("no", "n", "bye", "exit", "quit"):
                    break
                    
                # Extract fields from the conversation
                conversation_snippet = f"Assistant: {assistant_msg}\nUser: {user_input}"
                print("\nDEBUG: Before extraction:")
                print(f"DEBUG: Collected data: {self.collected_data}")
                
                self.extract_all_fields_working_copy(conversation_snippet)
                
                print("\nDEBUG: After extraction:")
                print(f"DEBUG: Collected data: {self.collected_data}")

                # Check if everything is collected
                missing = self.get_missing_fields()
                if not missing:
                    # All fields collected, store the initiative
                    all_initiatives.append(self.collected_data.copy())
                    # Update CSV with new initiative
                    self.update_initiatives_csv(self.collected_data.copy())
                    break

                # Update current status and next field
                current_status = self.format_current_status()
                next_field = missing[0] if missing else None
                
                print(f"\nDEBUG: Current status: {current_status}")
                print(f"DEBUG: Missing fields: {missing}")
                print(f"DEBUG: Next field to collect: {next_field}")

                # Create next prompt
                next_prompt = self.chat_prompt.format_messages(
                    initial_context=self.initial_context,
                    current_status=current_status,
                    missing_fields=", ".join(missing),
                    history=messages,
                    input=user_input,
                    previous_initiatives_context=self.previous_initiatives_context
                )
                
                response = self.llm.invoke(next_prompt)
                assistant_msg = response.content.strip()
                
                # Print thoughts separately
                print("\nDEBUG: Assistant's thoughts:")
                thoughts = assistant_msg.split("Thought")
                for thought in thoughts[1:]:  # Skip the first split which is empty
                    print(f"DEBUG: Thought{thought.strip()}")
                
                # Print the final answer
                answer = thoughts[-1].split("Answer:")[-1].strip()
                print(f"\nAssistant: {answer}")
                self.transcript.append(f"Assistant: {assistant_msg}")
                
                messages.append(HumanMessage(content=user_input))
                messages.append(AIMessage(content=assistant_msg))

        # Display and save collected data
        if all_initiatives:
            print("All collected AI initiatives:")
            df = pd.DataFrame(all_initiatives)
            
            # Clean up the display
            pd.set_option('display.max_colwidth', None)
            pd.set_option('display.width', None)
            
            # Format specific columns
            if 'Budget' in df.columns:
                df['Budget'] = df['Budget'].apply(lambda x: self._clean_budget(x) if pd.notnull(x) else x)
            if 'Success Metrics' in df.columns:
                df['Success Metrics'] = df['Success Metrics'].apply(
                    lambda x: self._clean_success_metrics(x) if pd.notnull(x) else x
                )
            if 'Tech Stack' in df.columns:
                # Convert lists to strings before cleaning
                df['Tech Stack'] = df['Tech Stack'].apply(
                    lambda x: self._clean_tech_stack(x) if pd.notnull(x) else x
                )
            if 'Department' in df.columns:
                df['Department'] = df['Department'].apply(lambda x: self._clean_department(x) if pd.notnull(x) else x)
            
            # Remove Next Steps if it contains the last question
            if 'Next Steps' in df.columns:
                df = df.drop('Next Steps', axis=1)
            
            print(df.to_string(index=False))
            
            # Save to CSV file
            try:
                # Save initiatives data
                filename = "C:\\Misc\\Mphasis\\Mphasis.AI\\AI_Initiatives.csv"
                df.to_csv(filename, index=False)
                print(f"\nData has been saved to {filename}")
                
                # Save transcript
                transcript_filename = "C:\\Misc\\Mphasis\\Mphasis.AI\\Conversation_Transcript.txt"
                with open(transcript_filename, 'w', encoding='utf-8') as f:
                    # Add timestamp and header
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"Conversation Transcript - {timestamp}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # Write each interaction
                    for line in self.transcript:
                        f.write(line + "\n")
                    
                    # Add summary of collected data
                    f.write("\n" + "=" * 50 + "\n")
                    f.write("Collected Data Summary:\n")
                    f.write("=" * 50 + "\n")
                    f.write(df.to_string(index=False))
                
                print(f"Conversation transcript has been saved to {transcript_filename}")
            except Exception as e:
                print(f"\nError saving files: {str(e)}")
        else:
            print("No initiatives were collected.")

    def load_previous_initiatives(self):
        """Load previously collected initiatives from Azure Blob CSV file."""
        try:
            # Use Azure Blob Storage URL for AI_Initiatives.csv like other CSV files
            sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN", "")
            base_url = "https://acu1tmagentbotd1saglobal.blob.core.windows.net/csvdata"
            
            if sas_token:
                initiatives_csv_url = f"{base_url}/AI_Initiatives.csv?{sas_token}"
            else:
                initiatives_csv_url = os.getenv("INITIATIVES_CSV_URL", f"{base_url}/AI_Initiatives.csv")
            
            # Try to read from Azure Blob
            initiatives_response = requests.get(initiatives_csv_url, timeout=30)
            if initiatives_response.status_code == 200:
                initiatives_csv = StringIO(initiatives_response.text)
                df = pd.read_csv(initiatives_csv)
                print(f"Loaded {len(df)} previous initiatives from Azure Blob")
                return df.to_dict('records')  # Convert to list of dictionaries
            else:
                print("No previous initiatives file found in Azure Blob")
                return []
        except Exception as e:
            print(f"Error loading previous initiatives from Azure Blob: {str(e)}")
            return []

    def format_previous_initiatives_context(self, previous_initiatives):
        """Format previous initiatives into a context string."""
        if not previous_initiatives:
            return "No previous initiatives found."
        
        context_lines = ["Previous AI Initiatives:"]
        for i, initiative in enumerate(previous_initiatives, 1):
            context_lines.append(f"\nInitiative {i}:")
            for field, value in initiative.items():
                if pd.notna(value):  # Only include non-null values
                    context_lines.append(f"  {field}: {value}")
        
        return "\n".join(context_lines)

    # Update format_messages to include previous initiatives
    def format_messages(self, **kwargs):
        return self.chat_prompt.format_messages(
            previous_initiatives_context=self.previous_initiatives_context,
            **kwargs
        )

# Create and run the survey agent
if __name__ == "__main__":
    agent = SurveyAgent()
    agent.run_survey()
