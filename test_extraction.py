#!/usr/bin/env python3

def test_extraction_logic():
    """Test the extraction logic to ensure we only get the Answer, not all thoughts"""
    
    # Simulate LLM response with all thoughts (what we currently get)
    full_llm_response = """Thought 1: From the initial context, I see this is a conversation with a Customer Account Rep for HPE in the Technology sector. They mentioned working on "chat HWE" which appears to be their AI initiative name. I should now ask about the type of AI technology they're using in this project.

Thought 2: Looking at previous initiatives, there are no previous initiatives in the CSV file to reference. I should focus on understanding the current initiative without making assumptions about past work.

Thought 3: The next missing field is "Type of AI" which is crucial for understanding the technical nature of their initiative. This will help determine the complexity, resources needed, and potential challenges they might face.

Thought 4: Since they mentioned "chat HWE", this suggests it might be a conversational AI or chatbot project. I should ask specifically about the AI technology stack to understand if it's using large language models, traditional NLP, or other AI approaches.

Thought 5: I should craft a question that builds on their "chat HWE" mention and specifically asks about the AI technology type while showing understanding of their business context.

Answer: Thank you for sharing that you're working on the "chat HWE" initiative. Could you tell me more about the specific type of AI technology you're using for this project? For example, are you leveraging large language models, traditional natural language processing, machine learning algorithms, or a combination of different AI approaches?"""

    # Test current extraction logic
    def extract_answer_current(assistant_msg):
        """Current extraction logic from survey_api.py"""
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
        return answer

    # Test improved extraction logic
    def extract_answer_improved(assistant_msg):
        """Improved extraction logic"""
        # First try to find "Answer:" section
        if "Answer:" in assistant_msg:
            answer = assistant_msg.split("Answer:")[-1].strip()
            return answer
        
        # If no Answer section, try to find the last question in the response
        if "?" in assistant_msg:
            # Split by sentences and find the last question
            sentences = [s.strip() for s in assistant_msg.replace('\n', ' ').split('.') if s.strip()]
            for sentence in reversed(sentences):
                if "?" in sentence:
                    return sentence.strip()
        
        # Fallback: return the last non-empty line
        lines = [line.strip() for line in assistant_msg.split('\n') if line.strip()]
        if lines:
            return lines[-1]
        
        return assistant_msg

    # Test both approaches
    print("=== Testing Extraction Logic ===\n")
    
    print("Full LLM Response (what bot currently speaks):")
    print(f"Length: {len(full_llm_response)} characters")
    print(f"Preview: {full_llm_response[:200]}...\n")
    
    current_result = extract_answer_current(full_llm_response)
    print("Current Logic Result:")
    print(f"Length: {len(current_result)} characters")
    print(f"Result: {current_result}\n")
    
    improved_result = extract_answer_improved(full_llm_response)
    print("Improved Logic Result:")
    print(f"Length: {len(improved_result)} characters") 
    print(f"Result: {improved_result}\n")

    # Test edge cases
    print("=== Edge Case Tests ===\n")
    
    # Case 1: No Answer section
    no_answer = """Thought 1: Analysis here.
Thought 2: More analysis.
What specific AI technology are you using?"""
    
    print("Test 1 - No Answer section:")
    print(f"Current: {extract_answer_current(no_answer)}")
    print(f"Improved: {extract_answer_improved(no_answer)}\n")
    
    # Case 2: Multiple questions
    multiple_q = """Answer: I have a question about your project. What type of AI are you using? Is it machine learning based?"""
    
    print("Test 2 - Multiple questions:")
    print(f"Current: {extract_answer_current(multiple_q)}")
    print(f"Improved: {extract_answer_improved(multiple_q)}\n")

if __name__ == "__main__":
    test_extraction_logic()