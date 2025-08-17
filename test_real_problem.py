#!/usr/bin/env python3

def test_real_problem():
    """Test the actual scenario that causes the bot to speak all thoughts"""
    
    # This is the scenario that would trigger line 177: answer = assistant_msg
    # When there's no "Answer:" AND no "Thought 5:" sections
    malformed_response = """I need to gather information about your AI initiative.

Let me analyze your previous response about the chat project.

Based on your business context at HPE, I should ask about the technical details.

What specific type of artificial intelligence technology are you implementing in this project?"""

    # Another problematic case - when LLM doesn't follow the format
    non_formatted_response = """Thank you for sharing details about your chat project. I'd like to understand more about the technology stack you're using. Could you elaborate on whether you're using natural language processing, machine learning models, or other AI technologies for this initiative?"""

    def current_logic(assistant_msg):
        if "Answer:" in assistant_msg:
            return assistant_msg.split("Answer:")[-1].strip()
        elif "Thought 5:" in assistant_msg:
            thought5_section = assistant_msg.split("Thought 5:")[-1].strip()
            if "?" in thought5_section:
                sentences = thought5_section.split(".")
                for sentence in reversed(sentences):
                    if "?" in sentence:
                        return sentence.strip()
                        break
                else:
                    return thought5_section
            else:
                return thought5_section
        else:
            return assistant_msg  # ❌ PROBLEM: returns entire response

    def simple_fix(assistant_msg):
        # Try Answer: section first
        if "Answer:" in assistant_msg:
            return assistant_msg.split("Answer:")[-1].strip()
        
        # Try Thought 5: section
        if "Thought 5:" in assistant_msg:
            thought5_section = assistant_msg.split("Thought 5:")[-1].strip()
            if "?" in thought5_section:
                sentences = thought5_section.split(".")
                for sentence in reversed(sentences):
                    if "?" in sentence:
                        return sentence.strip()
            return thought5_section
        
        # NEW: If neither section exists, find the last question
        if "?" in assistant_msg:
            lines = assistant_msg.split('\n')
            for line in reversed(lines):
                if "?" in line and line.strip():
                    return line.strip()
        
        # Final fallback: last non-empty line
        lines = [line.strip() for line in assistant_msg.split('\n') if line.strip()]
        return lines[-1] if lines else assistant_msg

    print("=== Testing Real Problem Cases ===\n")
    
    print("Test 1 - Malformed response (no Answer: or Thought 5:):")
    print(f"Input length: {len(malformed_response)} chars")
    print(f"Current result length: {len(current_logic(malformed_response))} chars")
    print(f"Fixed result length: {len(simple_fix(malformed_response))} chars")
    print(f"Current: {current_logic(malformed_response)}")
    print(f"Fixed: {simple_fix(malformed_response)}\n")
    
    print("Test 2 - Non-formatted response:")
    print(f"Input length: {len(non_formatted_response)} chars") 
    print(f"Current result length: {len(current_logic(non_formatted_response))} chars")
    print(f"Fixed result length: {len(simple_fix(non_formatted_response))} chars")
    print(f"Current: {current_logic(non_formatted_response)}")
    print(f"Fixed: {simple_fix(non_formatted_response)}")

if __name__ == "__main__":
    test_real_problem()