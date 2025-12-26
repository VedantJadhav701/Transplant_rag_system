#!/usr/bin/env python3
"""
API Test Script
===============

Test the FastAPI endpoints including authentication.

Usage:
    python test_api.py
"""

import requests
import json
from pprint import pprint

BASE_URL = "http://localhost:8000/api/v1"


def test_authentication():
    """Test authentication endpoint"""
    print("\n" + "="*80)
    print("Testing /token endpoint (Authentication)...")
    print("="*80)
    
    # Test with valid credentials
    payload = {
        "username": "admin@transplant.ai",
        "password": "admin123"
    }
    
    response = requests.post(f"{BASE_URL}/token", json=payload)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Authentication successful!")
        print(f"Token type: {data['token_type']}")
        print(f"Expires in: {data['expires_in']} seconds")
        print(f"Access token: {data['access_token'][:50]}...")
        return data['access_token']
    else:
        print(f"❌ Authentication failed: {response.json()}")
        return None


def test_invalid_authentication():
    """Test with invalid credentials"""
    print("\n" + "="*80)
    print("Testing authentication with invalid credentials...")
    print("="*80)
    
    payload = {
        "username": "wrong@example.com",
        "password": "wrongpassword"
    }
    
    response = requests.post(f"{BASE_URL}/token", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Expected error: {response.json()['detail']}")


def test_health():
    """Test health endpoint"""
    print("\n" + "="*80)
    print("Testing /health endpoint...")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    pprint(response.json())


def test_query_without_auth():
    """Test protected endpoint without authentication"""
    print("\n" + "="*80)
    print("Testing /query without authentication...")
    print("="*80)
    
    payload = {
        "query": "What are immunosuppressive drugs?",
        "top_k": 5
    }
    
    response = requests.post(f"{BASE_URL}/query", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Expected error: {response.json()['detail']}")


def test_query_with_auth(token: str, query: str, top_k: int = 5):
    """Test protected query endpoint with authentication"""
    print("\n" + "="*80)
    print(f"Testing /query endpoint with authentication...")
    print(f"Query: {query}")
    print("="*80)
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "query": query,
        "top_k": top_k,
        "max_tokens": 512,
        "model": "phi3:mini"
    }
    
    response = requests.post(
        f"{BASE_URL}/query",
        json=payload,
        headers=headers
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"\n📌 ANSWER (Confidence: {data['confidence']} - {data['confidence_score']}):")
        print("-" * 80)
        print(data['answer'][:500] + "..." if len(data['answer']) > 500 else data['answer'])
        print("-" * 80)
        
        print(f"\n📚 SOURCES ({len(data['sources'])} documents):")
        for i, source in enumerate(data['sources'], 1):
            print(f"\n[{i}] {source['document'][:60]}...")
            print(f"    Section: {source['section'][:50]}...")
            print(f"    Relevance: {source['similarity_score']}")
        
        print(f"\n⏱️  TIMING:")
        print(f"    Retrieval: {data['retrieval_time']}s")
        print(f"    Generation: {data['generation_time']}s")
        print(f"    Total: {data['total_time']}s")
        print(f"    Tokens: {data['total_tokens']}")
        
        if data.get('user'):
            print(f"\n👤 User: {data['user']}")
    else:
        print("Error:")
        pprint(response.json())


def test_validation():
    """Test input validation"""
    print("\n" + "="*80)
    print("Testing input validation...")
    print("="*80)
    
    # Get token first
    token_response = requests.post(
        f"{BASE_URL}/token",
        json={"username": "admin@transplant.ai", "password": "admin123"}
    )
    token = token_response.json()["access_token"]
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test prohibited query
    payload = {
        "query": "What dose should I take?",
        "top_k": 5
    }
    
    response = requests.post(f"{BASE_URL}/query", json=payload, headers=headers)
    print(f"Status Code: {response.status_code}")
    if response.status_code != 200:
        print(f"Expected validation error: {response.json()['detail']}")


if __name__ == "__main__":
    print("\n🧪 Medical RAG API Tests (With Authentication)")
    print("="*80)
    print("Make sure the API is running: uvicorn app.main:app --reload")
    print("="*80)
    
    try:
        # Test 1: Authentication
        token = test_authentication()
        
        if not token:
            print("\n❌ Authentication failed. Cannot proceed with protected endpoint tests.")
            exit(1)
        
        # Test 2: Invalid authentication
        test_invalid_authentication()
        
        # Test 3: Health check (public endpoint)
        test_health()
        
        # Test 4: Protected endpoint without auth
        test_query_without_auth()
        
        # Test 5: Protected endpoint with auth
        test_query_with_auth(token, "What are signs of acute kidney rejection?")
        
        # Test 6: Another query
        test_query_with_auth(token, "What are immunosuppressive drug mechanisms?", top_k=3)
        
        # Test 7: Input validation
        test_validation()
        
        print("\n" + "="*80)
        print("✅ All tests completed!")
        print("="*80)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to API")
        print("Start the server with: uvicorn app.main:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")
