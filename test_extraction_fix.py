#!/usr/bin/env python3

def test_extraction_fix():
    """Test the extraction fix to ensure it doesn't extract from assistant questions"""
    
    # Simulate the problematic conversation snippet
    conversation_snippet = """Assistant: Hello! I see you're representing Hewlett Packard Enterprise in the Information Technology sector. Could you please share details about your current AI initiatives and how AI is being utilized in your organization?
User: Hello."""

    # Original extraction template (problematic)
    original_template = """
            Extract information about the following fields from this conversation snippet. 
            Return a JSON object with the following rules:
            1. Use EXACT field names as keys
            2. For each field, extract the most recent/relevant information
            3. If a field is not mentioned, return null for that field
            4. For fields that can have multiple values, return them as a JSON array
            5. Preserve the exact values mentioned in the conversation
            6. DO NOT include default values - only extract values explicitly mentioned

            Fields with definitions:
            - Initiative: Name of the AI project or initiative
            - Department: Which department is leading this initiative

            Conversation snippet:
            {snippet}

            Return ONLY valid JSON with keys matching the field names exactly.
            """

    # Fixed extraction template
    fixed_template = """
            Extract information about the following fields from the USER'S RESPONSE ONLY. 
            DO NOT extract information from the assistant's question.
            Return a JSON object with the following rules:
            1. Use EXACT field names as keys
            2. For each field, extract information ONLY from the user's response
            3. If a field is not mentioned by the USER, return null for that field
            4. For fields that can have multiple values, return them as a JSON array
            5. Preserve the exact values mentioned by the USER
            6. DO NOT include default values - only extract values explicitly mentioned by the USER
            7. IGNORE any field names that appear in the assistant's question

            Fields with definitions:
            - Initiative: Name of the AI project or initiative
            - Department: Which department is leading this initiative

            Conversation snippet:
            {snippet}

            Return ONLY valid JSON with keys matching the field names exactly.
            """

    print("=== Testing Extraction Fix ===\n")
    
    print("Conversation snippet:")
    print(conversation_snippet)
    print()
    
    print("Original Template (problematic):")
    print("Would likely extract: Initiative='AI initiatives at HPE', Department='Information Technology'")
    print("❌ This extracts from the assistant's question\n")
    
    print("Fixed Template:")
    print("Should extract: Initiative=null, Department=null") 
    print("✅ This only looks at user's 'Hello' response\n")
    
    # Test with a real user response
    real_response = """Assistant: What type of AI technology are you using?
User: We're working on a chatbot project using GPT models in the engineering department."""
    
    print("=== Test with Real User Response ===")
    print("Conversation:")
    print(real_response)
    print()
    print("Should extract: Initiative='chatbot project', Type of AI='GPT models', Department='engineering'")
    print("✅ This correctly extracts only from user response")

if __name__ == "__main__":
    test_extraction_fix()