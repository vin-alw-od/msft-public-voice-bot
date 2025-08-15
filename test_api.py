#!/usr/bin/env python3
"""
Test script for Survey Agent FastAPI
This script tests the API endpoints to ensure they work correctly
"""

import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_start_survey():
    """Test starting a new survey session"""
    print("\n🔍 Testing start survey endpoint...")
    try:
        payload = {
            "user_id": "test_user_123",
            "context": {"test": "data"}
        }
        response = requests.post(f"{API_BASE_URL}/start-survey", 
                               headers=HEADERS, 
                               data=json.dumps(payload))
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Start survey passed")
            print(f"   Session ID: {data['session_id']}")
            print(f"   Message: {data['message'][:100]}...")
            return data['session_id']
        else:
            print(f"❌ Start survey failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Start survey error: {e}")
        return None

def test_process_input(session_id, user_input):
    """Test processing user input"""
    print(f"\n🔍 Testing process input: '{user_input}'...")
    try:
        payload = {
            "session_id": session_id,
            "user_input": user_input
        }
        response = requests.post(f"{API_BASE_URL}/process-input", 
                               headers=HEADERS, 
                               data=json.dumps(payload))
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Process input passed")
            print(f"   Status: {data['status']}")
            print(f"   Message: {data['message'][:100]}...")
            if data.get('missing_fields'):
                print(f"   Missing fields: {data['missing_fields'][:3]}...")
            return data
        else:
            print(f"❌ Process input failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Process input error: {e}")
        return None

def test_session_status(session_id):
    """Test getting session status"""
    print(f"\n🔍 Testing session status...")
    try:
        response = requests.get(f"{API_BASE_URL}/session/{session_id}/status")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Session status passed")
            print(f"   Status: {data['status']}")
            print(f"   Missing fields: {len(data['missing_fields'])} remaining")
            return data
        else:
            print(f"❌ Session status failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Session status error: {e}")
        return None

def test_full_conversation():
    """Test a full conversation flow"""
    print("\n" + "="*60)
    print("🚀 Starting Full Conversation Test")
    print("="*60)
    
    # Test health check first
    if not test_health_check():
        print("❌ Health check failed, aborting tests")
        return False
    
    # Start survey
    session_id = test_start_survey()
    if not session_id:
        print("❌ Failed to start survey, aborting tests")
        return False
    
    # Test conversation with sample inputs
    test_inputs = [
        "Yes, we have a new AI initiative called 'Smart Customer Service'",
        "It's a chatbot for customer support using natural language processing",
        "We're currently in the planning stage",
        "The budget is around $500,000",
        "We want to reduce response time by 50% and improve customer satisfaction",
        "This will be managed by the IT department",
        "We're planning to use Python, TensorFlow, and Azure OpenAI",
        "bye"  # End the conversation
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n--- Turn {i} ---")
        result = test_process_input(session_id, user_input)
        
        if not result:
            print("❌ Conversation failed, stopping test")
            return False
            
        # Check if conversation is completed
        if result['status'] == 'completed':
            print("🎉 Conversation completed successfully!")
            break
            
        # Small delay between inputs
        time.sleep(0.5)
    
    print("\n✅ Full conversation test completed!")
    return True

def test_list_sessions():
    """Test listing active sessions"""
    print("\n🔍 Testing list sessions endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/sessions")
        if response.status_code == 200:
            data = response.json()
            print("✅ List sessions passed")
            print(f"   Active sessions: {data['active_sessions']}")
            return True
        else:
            print(f"❌ List sessions failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ List sessions error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Survey Agent API Test Suite")
    print("="*60)
    
    print("\n⚠️  Make sure the FastAPI server is running:")
    print("   python survey_api.py")
    print("   or")
    print("   uvicorn survey_api:app --reload")
    
    input("\nPress Enter when the server is ready...")
    
    # Run individual endpoint tests first
    print("\n📋 Running Individual Endpoint Tests...")
    
    # Test basic endpoints
    test_health_check()
    test_list_sessions()
    
    # Run full conversation test
    success = test_full_conversation()
    
    # Final summary
    print("\n" + "="*60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ FastAPI integration is working correctly")
        print("✅ Ready for .NET bot integration")
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Check the server logs for details")
    print("="*60)

if __name__ == "__main__":
    main()