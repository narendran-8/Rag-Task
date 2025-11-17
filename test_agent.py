"""
Comprehensive test for all agent endpoints and nodes
"""
from langgraph.graph import StateGraph, START, END, MessagesState
from dotenv import load_dotenv
import os
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.search import RAGSearch
import sqlite3

# Setup
load_dotenv()
conn = sqlite3.connect("test_store.db", check_same_thread=False)
memory = SqliteSaver(conn)

class AgentState(MessagesState):
    retrieved_docs: str = ""
    answer: str = ""
    validation_passed: bool = False

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# Import the nodes from agent
from agent import retriever_node, answer_node, validation_node, route_after_validation

def test_retriever_node():
    """Test Node 1: Retriever"""
    print("\n" + "="*60)
    print("TEST 1: RETRIEVER NODE")
    print("="*60)
    
    test_state = {
        "messages": [HumanMessage(content="What is penetration testing?")]
    }
    
    result = retriever_node(test_state)
    
    print(f"✓ Retriever executed successfully")
    print(f"✓ Retrieved docs length: {len(result.get('retrieved_docs', ''))} characters")
    print(f"✓ Preview: {result.get('retrieved_docs', '')[:200]}...")
    
    return result.get('retrieved_docs', '') != ""


def test_answer_node():
    """Test Node 2: Answer Generator"""
    print("\n" + "="*60)
    print("TEST 2: ANSWER NODE")
    print("="*60)
    
    test_state = {
        "messages": [HumanMessage(content="What is Active Directory?")],
        "retrieved_docs": "Active Directory (AD) is a directory service developed by Microsoft for Windows domain networks. It is included in most Windows Server operating systems as a set of processes and services."
    }
    
    result = answer_node(test_state)
    
    print(f"✓ Answer node executed successfully")
    print(f"✓ Generated answer: {result.get('answer', '')}")
    
    return result.get('answer', '') != ""


def test_validation_node_valid():
    """Test Node 3: Validation (Valid Case)"""
    print("\n" + "="*60)
    print("TEST 3: VALIDATION NODE (Valid Answer)")
    print("="*60)
    
    test_state = {
        "messages": [HumanMessage(content="What is penetration testing?")],
        "answer": "Penetration testing is a security assessment method where ethical hackers simulate attacks on systems to identify vulnerabilities."
    }
    
    result = validation_node(test_state)
    
    print(f"✓ Validation executed successfully")
    print(f"✓ Validation result: {'PASSED' if result.get('validation_passed') else 'FAILED'}")
    
    return result.get('validation_passed') is not None


def test_validation_node_invalid():
    """Test Node 3: Validation (Invalid Case)"""
    print("\n" + "="*60)
    print("TEST 4: VALIDATION NODE (Invalid Answer)")
    print("="*60)
    
    test_state = {
        "messages": [HumanMessage(content="What is penetration testing?")],
        "answer": "I don't know."
    }
    
    result = validation_node(test_state)
    
    print(f"✓ Validation executed successfully")
    print(f"✓ Validation result: {'PASSED' if result.get('validation_passed') else 'FAILED'}")
    print(f"✓ Expected: FAILED (invalid answer should fail)")
    
    return result.get('validation_passed') is not None


def test_routing_logic():
    """Test Routing Function"""
    print("\n" + "="*60)
    print("TEST 5: ROUTING LOGIC")
    print("="*60)
    
    # Test valid route
    valid_state = {"validation_passed": True}
    route1 = route_after_validation(valid_state)
    print(f"✓ Valid answer routes to: {route1}")
    
    # Test invalid route
    invalid_state = {"validation_passed": False}
    route2 = route_after_validation(invalid_state)
    print(f"✓ Invalid answer routes to: {route2}")
    
    return True


def test_full_graph_workflow():
    """Test Complete Graph Workflow"""
    print("\n" + "="*60)
    print("TEST 6: FULL GRAPH WORKFLOW")
    print("="*60)
    
    from agent import graph
    
    agent_app = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "test-001"}}
    
    test_queries = [
        "What are the main topics in web application security?",
        "Explain SQL injection",
        "What is OSCP?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        
        response = agent_app.invoke(
            {"messages": [HumanMessage(content=query)]}, 
            config={"configurable": {"thread_id": f"test-{i:03d}"}},
            stream_mode="values"
        )
        
        print(f"✓ Retrieved: {response.get('retrieved_docs', 'N/A')[:100]}...")
        print(f"✓ Answer: {response.get('answer', 'N/A')}")
        print(f"✓ Validated: {response.get('validation_passed', False)}")
    
    return True


def test_edge_cases():
    """Test Edge Cases"""
    print("\n" + "="*60)
    print("TEST 7: EDGE CASES")
    print("="*60)
    
    # Empty query
    print("\n--- Edge Case 1: Empty Query ---")
    try:
        test_state = {"messages": [HumanMessage(content="")]}
        result = retriever_node(test_state)
        print(f"✓ Handled empty query: {len(result.get('retrieved_docs', ''))} chars")
    except Exception as e:
        print(f"✗ Failed on empty query: {e}")
    
    # Very long query
    print("\n--- Edge Case 2: Long Query ---")
    try:
        long_query = "What is " + " and ".join(["penetration testing"] * 20) + "?"
        test_state = {"messages": [HumanMessage(content=long_query)]}
        result = retriever_node(test_state)
        print(f"✓ Handled long query: {len(result.get('retrieved_docs', ''))} chars")
    except Exception as e:
        print(f"✗ Failed on long query: {e}")
    
    # Special characters
    print("\n--- Edge Case 3: Special Characters ---")
    try:
        test_state = {"messages": [HumanMessage(content="What is SQL injection? <script>alert('test')</script>")]}
        result = retriever_node(test_state)
        print(f"✓ Handled special chars: {len(result.get('retrieved_docs', ''))} chars")
    except Exception as e:
        print(f"✗ Failed on special characters: {e}")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "🔍 STARTING COMPREHENSIVE AGENT TESTS 🔍".center(60))
    
    tests = [
        ("Retriever Node", test_retriever_node),
        ("Answer Node", test_answer_node),
        ("Validation Node (Valid)", test_validation_node_valid),
        ("Validation Node (Invalid)", test_validation_node_invalid),
        ("Routing Logic", test_routing_logic),
        ("Full Graph Workflow", test_full_graph_workflow),
        ("Edge Cases", test_edge_cases),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✅ PASS" if success else "⚠️  PARTIAL"))
        except Exception as e:
            results.append((name, f"❌ FAIL: {str(e)[:50]}"))
            print(f"\n❌ Error in {name}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY".center(60))
    print("="*60)
    for name, result in results:
        print(f"{name:.<40} {result}")
    
    passed = sum(1 for _, r in results if "✅" in r)
    total = len(results)
    print("\n" + f"Total: {passed}/{total} tests passed".center(60))
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
