#!/usr/bin/env python3
"""
Performance Benchmarking Tool
==============================
Measure latency, throughput, and resource usage under load.
"""

import time
import statistics
import concurrent.futures
from typing import List, Dict
import requests
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Single query benchmark result"""
    latency: float
    success: bool
    error: str = None


class RAGBenchmark:
    """Benchmark RAG system performance"""
    
    def __init__(self, api_url: str = "http://localhost:8000/api/v1"):
        self.api_url = api_url
        self.token = None
    
    def authenticate(self, username: str = "admin@transplant.ai", password: str = "admin123"):
        """Get JWT token"""
        response = requests.post(
            f"{self.api_url}/token",
            json={"username": username, "password": password}
        )
        self.token = response.json()["access_token"]
    
    def single_query(self, question: str) -> BenchmarkResult:
        """Execute single query and measure latency"""
        start = time.time()
        try:
            response = requests.post(
                f"{self.api_url}/query",
                json={"query": question},
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=120
            )
            latency = time.time() - start
            success = response.status_code == 200
            return BenchmarkResult(latency=latency, success=success)
        except Exception as e:
            return BenchmarkResult(latency=time.time() - start, success=False, error=str(e))
    
    def latency_test(self, questions: List[str], iterations: int = 10) -> Dict:
        """Test latency with sequential requests"""
        print(f"\n⏱️  Latency Test ({iterations} iterations)")
        print("="*60)
        
        results = []
        for i in range(iterations):
            question = questions[i % len(questions)]
            result = self.single_query(question)
            results.append(result)
            print(f"  Query {i+1}: {result.latency:.3f}s {'✓' if result.success else '✗'}")
        
        successes = [r.latency for r in results if r.success]
        
        if not successes:
            print("❌ All queries failed!")
            return {}
        
        metrics = {
            "total_queries": iterations,
            "successful": len(successes),
            "failed": len(results) - len(successes),
            "success_rate": len(successes) / len(results),
            "avg_latency": statistics.mean(successes),
            "median_latency": statistics.median(successes),
            "min_latency": min(successes),
            "max_latency": max(successes),
            "p95_latency": sorted(successes)[int(len(successes) * 0.95)] if len(successes) > 1 else successes[0],
            "p99_latency": sorted(successes)[int(len(successes) * 0.99)] if len(successes) > 1 else successes[0],
        }
        
        print(f"\n📊 Results:")
        print(f"  Success Rate: {metrics['success_rate']*100:.1f}%")
        print(f"  Avg Latency:  {metrics['avg_latency']:.3f}s")
        print(f"  Median:       {metrics['median_latency']:.3f}s")
        print(f"  Min/Max:      {metrics['min_latency']:.3f}s / {metrics['max_latency']:.3f}s")
        print(f"  P95:          {metrics['p95_latency']:.3f}s")
        
        return metrics
    
    def throughput_test(self, questions: List[str], concurrent_users: int = 3, duration: int = 30) -> Dict:
        """Test throughput with concurrent requests"""
        print(f"\n🚀 Throughput Test ({concurrent_users} concurrent users, {duration}s)")
        print("="*60)
        
        start_time = time.time()
        completed = 0
        errors = 0
        latencies = []
        
        def worker():
            nonlocal completed, errors
            while time.time() - start_time < duration:
                question = questions[completed % len(questions)]
                result = self.single_query(question)
                if result.success:
                    completed += 1
                    latencies.append(result.latency)
                else:
                    errors += 1
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(worker) for _ in range(concurrent_users)]
            concurrent.futures.wait(futures)
        
        elapsed = time.time() - start_time
        
        metrics = {
            "duration": elapsed,
            "concurrent_users": concurrent_users,
            "total_completed": completed,
            "total_errors": errors,
            "queries_per_second": completed / elapsed,
            "avg_latency": statistics.mean(latencies) if latencies else 0,
        }
        
        print(f"\n📊 Results:")
        print(f"  Duration:      {metrics['duration']:.1f}s")
        print(f"  Completed:     {metrics['total_completed']} queries")
        print(f"  Errors:        {metrics['total_errors']}")
        print(f"  Throughput:    {metrics['queries_per_second']:.2f} queries/sec")
        print(f"  Avg Latency:   {metrics['avg_latency']:.3f}s")
        
        return metrics


def main():
    """Run benchmarks"""
    print("🔬 RAG System Performance Benchmark")
    print("="*60)
    
    # Test questions
    questions = [
        "What is acute rejection?",
        "Explain HLA antibodies",
        "What is tacrolimus mechanism?",
        "Signs of kidney rejection?",
        "What is crossmatch test?",
    ]
    
    # Initialize
    bench = RAGBenchmark()
    print("\n🔐 Authenticating...")
    bench.authenticate()
    print("✅ Authenticated")
    
    # Warm-up
    print("\n🔥 Warming up...")
    bench.single_query(questions[0])
    print("✅ Warm-up complete")
    
    # Run latency test
    latency_metrics = bench.latency_test(questions, iterations=10)
    
    # Run throughput test
    throughput_metrics = bench.throughput_test(questions, concurrent_users=2, duration=30)
    
    print("\n" + "="*60)
    print("✅ Benchmark complete!")
    print("="*60)


if __name__ == "__main__":
    main()
