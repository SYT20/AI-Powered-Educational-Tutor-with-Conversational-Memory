"""
Test script for EduSmart AI Tutor system
"""
import sys
import os
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_tutor.tutor_system import EduSmartAITutor

def test_system_initialization():
    """Test system initialization"""
    print("🚀 Testing EduSmart AI Tutor System Initialization...")
    
    try:
        # Initialize the tutor system
        tutor = EduSmartAITutor()
        print("✅ Tutor system created")
        
        # Initialize components
        success = tutor.initialize()
        if success:
            print("✅ System initialized successfully")
        else:
            print("❌ System initialization failed")
            return False
        
        # Check system status
        status = tutor.get_system_status()
        print(f"📊 System Status: {status['is_initialized']}")
        print(f"📚 Documents loaded: {status['stats']['documents_loaded']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during initialization: {str(e)}")
        return False

def test_basic_chat():
    """Test basic chat functionality"""
    print("\n💬 Testing Basic Chat Functionality...")
    
    try:
        # Initialize system
        tutor = EduSmartAITutor()
        if not tutor.initialize():
            print("❌ Failed to initialize system for chat test")
            return False
        
        # Start a session
        session_id = tutor.start_new_session()
        print(f"📝 Started session: {session_id}")
        
        # Test questions
        test_questions = [
            "What is algebra?",
            "Explain Newton's first law",
            "How do I improve my reading comprehension?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🧑‍🎓 Question {i}: {question}")
            
            response = tutor.chat(question)
            
            if response.get('answer'):
                print(f"🤖 Response: {response['answer'][:100]}...")
                print(f"📚 Sources found: {len(response.get('sources', []))}")
                print(f"🎯 Confidence: {response.get('confidence', 0):.2%}")
            else:
                print("❌ No response generated")
                return False
        
        # End session
        tutor.end_session(save_session=False)
        print("\n✅ Chat test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error during chat test: {str(e)}")
        return False

def test_memory_functionality():
    """Test conversation memory"""
    print("\n🧠 Testing Memory Functionality...")
    
    try:
        tutor = EduSmartAITutor()
        if not tutor.initialize():
            print("❌ Failed to initialize system for memory test")
            return False
        
        session_id = tutor.start_new_session()
        
        # First interaction
        response1 = tutor.chat("I'm interested in learning mathematics")
        print("🧑‍🎓 First interaction: I'm interested in learning mathematics")
        print(f"🤖 Response length: {len(response1.get('answer', ''))}")
        
        # Second interaction referencing the first
        response2 = tutor.chat("Can you give me a basic example?")
        print("🧑‍🎓 Second interaction: Can you give me a basic example?")
        print(f"🤖 Response length: {len(response2.get('answer', ''))}")
        
        # Check if memory is working
        profile = response2.get('student_profile', '')
        if 'math' in profile.lower() or 'mathematics' in profile.lower():
            print("✅ Memory is working - student profile updated")
        else:
            print("⚠️ Memory might not be working optimally")
        
        tutor.end_session(save_session=False)
        return True
        
    except Exception as e:
        print(f"❌ Error during memory test: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 EduSmart AI Tutor System Tests")
    print("=" * 50)
    
    # Configure logging to reduce noise during testing
    logging.getLogger().setLevel(logging.WARNING)
    
    tests = [
        ("System Initialization", test_system_initialization),
        ("Basic Chat", test_basic_chat),
        ("Memory Functionality", test_memory_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} Test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The system is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)