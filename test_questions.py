"""
Five comprehensive test questions for the RAG system
Testing different aspects of penetration testing knowledge
"""
from langchain_core.messages import HumanMessage
from agent import graph
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Setup
conn = sqlite3.connect("questions_test.db", check_same_thread=False)
memory = SqliteSaver(conn)

# Compile the graph
agent_app = graph.compile(checkpointer=memory)

# Five test questions covering different aspects
test_questions = [
    {
        "id": 1,
        "question": "What are the key techniques used in Active Directory enumeration and attacks?",
        "category": "Active Directory Security"
    },
    {
        "id": 2,
        "question": "Explain the difference between SQL injection and XSS attacks, including their impact and prevention methods.",
        "category": "Web Application Vulnerabilities"
    },
    {
        "id": 3,
        "question": "What tools and methodologies are recommended for vulnerability scanning in penetration testing?",
        "category": "Vulnerability Assessment"
    },
    {
        "id": 4,
        "question": "Describe the post-exploitation phase in penetration testing and what activities are typically performed.",
        "category": "Post-Exploitation"
    },
    {
        "id": 5,
        "question": "What are the essential components that should be included in a penetration testing report?",
        "category": "Reporting & Documentation"
    }
]

def run_rag_questions():
    """Run all 5 test questions through the RAG system"""
    print("\n" + "="*80)
    print("RAG SYSTEM - 5 TEST QUESTIONS".center(80))
    print("="*80)
    
    results = []
    
    for test in test_questions:
        print(f"\n{'='*80}")
        print(f"QUESTION {test['id']}: {test['category']}")
        print(f"{'='*80}")
        print(f"\n📝 Question: {test['question']}\n")
        
        try:
            # Invoke the graph
            config = {"configurable": {"thread_id": f"q{test['id']:03d}"}}
            response = agent_app.invoke(
                {"messages": [HumanMessage(content=test['question'])]},
                config=config,
                stream_mode="values"
            )
            
            # Extract results
            retrieved_docs = response.get("retrieved_docs", "N/A")
            answer = response.get("answer", "N/A")
            validation_passed = response.get("validation_passed", False)
            
            # Display results
            print(f"📚 Retrieved Context Preview:")
            print(f"   {retrieved_docs[:200]}...")
            print(f"\n💡 Generated Answer:")
            print(f"   {answer}")
            print(f"\n✓ Validation Status: {'PASSED ✅' if validation_passed else 'FAILED ❌'}")
            
            # Store results
            results.append({
                "question_id": test['id'],
                "category": test['category'],
                "question": test['question'],
                "answer": answer,
                "validated": validation_passed,
                "context_length": len(retrieved_docs)
            })
            
        except Exception as e:
            print(f"\n❌ Error processing question {test['id']}: {e}")
            results.append({
                "question_id": test['id'],
                "category": test['category'],
                "question": test['question'],
                "answer": f"Error: {str(e)}",
                "validated": False,
                "context_length": 0
            })
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY REPORT".center(80))
    print(f"{'='*80}\n")
    
    for result in results:
        status = "✅ PASS" if result['validated'] else "❌ FAIL"
        print(f"Q{result['question_id']}: {result['category']:<35} {status}")
        print(f"     Context: {result['context_length']} chars | Answer: {len(result['answer'])} chars")
    
    passed = sum(1 for r in results if r['validated'])
    print(f"\n{'='*80}")
    print(f"Total Questions Validated: {passed}/{len(results)}".center(80))
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    print("\n🚀 Starting RAG System Test with 5 Questions...\n")
    results = run_rag_questions()
    print("\n✨ Test completed successfully!")
