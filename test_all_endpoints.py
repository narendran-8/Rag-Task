"""
Comprehensive test for all API endpoints
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def print_separator(title):
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)

def test_endpoint(endpoint, data, test_name):
    """Test a single endpoint"""
    print_separator(test_name)
    print(f"\nEndpoint: POST {endpoint}")
    print(f"Request Data: {json.dumps(data, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}{endpoint}", json=data, timeout=30)
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nResponse:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print(f"\n✓ SUCCESS")
            return True
        else:
            print(f"\n✗ FAILED")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"\n✗ ERROR: Cannot connect to server at {BASE_URL}")
        print("Make sure the server is running: uv run uvicorn main:app --reload")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        return False

def main():
    print_separator("TESTING ALL RAG ENDPOINTS")
    print("\nStarting comprehensive endpoint tests...")
    
    results = []
    
    # Test 1: Vector Search
    print("\n\nWaiting 1 second...")
    time.sleep(1)
    result = test_endpoint(
        "/Vector_Search",
        {"question": "What is penetration testing?"},
        "TEST 1: Vector Search (Raw Document Retrieval)"
    )
    results.append(("Vector Search", result))
    
    # Test 2: Direct RAG Question
    print("\n\nWaiting 2 seconds...")
    time.sleep(2)
    result = test_endpoint(
        "/Direct_RAG_Question",
        {"question": "What is SQL injection?"},
        "TEST 2: Direct RAG Question (Quick Search)"
    )
    results.append(("Direct RAG Question", result))
    
    # Test 3: RAG Bot (LangGraph)
    print("\n\nWaiting 2 seconds...")
    time.sleep(2)
    result = test_endpoint(
        "/rag_bot",
        {"question": "What is OSCP certification?"},
        "TEST 3: RAG Bot (LangGraph with Validation)"
    )
    results.append(("RAG Bot", result))
    
    # Test 4: Another RAG Bot question
    print("\n\nWaiting 2 seconds...")
    time.sleep(2)
    result = test_endpoint(
        "/rag_bot",
        {"question": "Explain Active Directory attacks"},
        "TEST 4: RAG Bot (Another Question)"
    )
    results.append(("RAG Bot #2", result))
    
    # Summary
    print_separator("TEST SUMMARY")
    passed = 0
    failed = 0
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "-"*80)
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("-"*80)
    
    if failed == 0:
        print("\n🎉 All endpoints are working correctly!")
    else:
        print(f"\n⚠️  {failed} endpoint(s) need attention")
    
    print("\n")

if __name__ == "__main__":
    main()
