"""Test Docker deployment of Medical RAG API"""
import requests
import json

BASE_URL = "http://localhost:8001/api/v1"

def test_docker_deployment():
    print("\n" + "="*80)
    print("🐳 DOCKER DEPLOYMENT TEST")
    print("="*80)
    
    # 1. Health Check
    print("\n1️⃣ Testing Health Endpoint...")
    health = requests.get(f"{BASE_URL}/health").json()
    print(f"   Status: {health['status']}")
    print(f"   ChromaDB: {'✅' if health['chroma_connected'] else '❌'}")
    print(f"   Ollama: {'✅' if health['model_available'] else '❌'}")
    print(f"   Chunks: {health['chunks_indexed']}")
    
    # 2. Authentication
    print("\n2️⃣ Testing Authentication...")
    auth_response = requests.post(
        f"{BASE_URL}/token",
        json={"username": "admin@transplant.ai", "password": "admin123"}
    )
    
    if auth_response.status_code == 200:
        token = auth_response.json()["access_token"]
        print(f"   ✅ Authentication successful")
        print(f"   Token: {token[:20]}...")
    else:
        print(f"   ❌ Authentication failed: {auth_response.status_code}")
        return
    
    # 3. Query Test
    print("\n3️⃣ Testing Query Endpoint...")
    headers = {"Authorization": f"Bearer {token}"}
    query_data = {
        "query": "What are the signs of acute kidney rejection?",
        "top_k": 3,
        "model": "gemma3:1b"
    }
    
    print(f"   Query: {query_data['query']}")
    print(f"   Model: {query_data['model']}")
    
    response = requests.post(
        f"{BASE_URL}/query",
        headers=headers,
        json=query_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n   ✅ Query successful!")
        print(f"   Confidence: {result.get('confidence_label', 'N/A')} ({result.get('confidence_score', 0):.3f})")
        print(f"   Answer length: {len(result.get('answer', ''))} chars")
        print(f"   Sources: {len(result.get('sources', []))} documents")
        print(f"   Timing: {result.get('total_time', 0):.2f}s")
        print(f"\n   📝 Answer preview:")
        answer = result.get('answer', 'No answer')
        print(f"   {answer[:200]}...")
    else:
        print(f"   ❌ Query failed: {response.status_code}")
        print(f"   {response.text}")
    
    print("\n" + "="*80)
    print("✅ Docker deployment test complete!")
    print("="*80)

if __name__ == "__main__":
    test_docker_deployment()
