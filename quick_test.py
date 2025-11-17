"""
Quick endpoint validation test
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoint(name, endpoint, data):
    print(f"\nTesting {name}...")
    try:
        response = requests.post(f"{BASE_URL}{endpoint}", json=data, timeout=20)
        if response.status_code == 200:
            print(f"✓ {name} - PASS")
            result = response.json()
            if "answer" in result:
                answer = str(result["answer"])
                print(f"  Answer preview: {answer[:100]}...")
            return True
        else:
            print(f"✗ {name} - FAIL (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"✗ {name} - ERROR: {str(e)[:50]}")
        return False

print("="*60)
print("QUICK ENDPOINT VALIDATION TEST")
print("="*60)

results = []

# Test 1
results.append(test_endpoint(
    "Vector Search",
    "/Vector_Search",
    {"question": "penetration testing"}
))

# Test 2
results.append(test_endpoint(
    "Direct RAG",
    "/Direct_RAG_Question",
    {"question": "What is web security?"}
))

# Test 3
results.append(test_endpoint(
    "RAG Bot",
    "/rag_bot",
    {"question": "What is the main goal of penetration testing?"}
))

print("\n" + "="*60)
print(f"RESULT: {sum(results)}/{len(results)} tests passed")
if all(results):
    print("✓ ALL ENDPOINTS WORKING!")
else:
    print("⚠ Some endpoints need attention")
print("="*60)
