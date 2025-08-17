#!/usr/bin/env python3

def test_simple_extraction_fix():
    """Test a simpler fix for the extraction logic"""
    
    # Current problematic case - no "Answer:" section
    no_answer_response = """Thought 1: Analysis of user response.
Thought 2: More detailed analysis here.
Thought 3: Field importance analysis.
Thought 4: Business context insights.
Thought 5: I should ask about the AI technology type.
What specific type of AI technology are you using for this chat project?"""

    # Current logic (problematic)
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
            return assistant_msg  # ❌ THIS IS THE PROBLEM - returns everything

    # Simple fix - just find the last question
    def simple_fix(assistant_msg):
        # First try Answer: section
        if "Answer:" in assistant_msg:
            return assistant_msg.split("Answer:")[-1].strip()
        
        # If no Answer section, find the last question in the entire response
        if "?" in assistant_msg:
            # Find all sentences with questions
            lines = assistant_msg.split('\n')
            for line in reversed(lines):
                if "?" in line and line.strip():
                    return line.strip()
        
        # Fallback: return last non-empty line
        lines = [line.strip() for line in assistant_msg.split('\n') if line.strip()]
        return lines[-1] if lines else assistant_msg

    print("=== Testing Simple Fix ===\n")
    
    print("Input (no Answer section):")
    print(f"{no_answer_response}\n")
    
    current_result = current_logic(no_answer_response)
    print("Current Logic Result:")
    print(f"Length: {len(current_result)} chars")
    print(f"Result: {current_result}\n")
    
    simple_result = simple_fix(no_answer_response)
    print("Simple Fix Result:")
    print(f"Length: {len(simple_result)} chars")
    print(f"Result: {simple_result}\n")
    
    # Test with Answer section (should work the same)
    with_answer = """Thought 1: Analysis here.
Answer: What type of AI are you using?"""
    
    print("=== Test with Answer section ===")
    print("Current:", current_logic(with_answer))
    print("Simple Fix:", simple_fix(with_answer))

if __name__ == "__main__":
    test_simple_extraction_fix()