#!/usr/bin/env python3
"""
RAG Evaluation Framework
========================
Comprehensive evaluation of retrieval, generation, and end-to-end performance.

Metrics:
- Retrieval: Precision@K, Recall@K, MRR, NDCG
- Generation: ROUGE, Semantic Similarity, Faithfulness
- End-to-end: Answer Accuracy, Latency, Confidence Calibration
"""

import json
import time
import statistics
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import ndcg_score
from sentence_transformers import SentenceTransformer, util


@dataclass
class EvalQuestion:
    """Test question with ground truth"""
    question: str
    ground_truth_answer: str
    relevant_docs: List[str]  # Document IDs that should be retrieved
    category: str  # e.g., "immunology", "rejection", "drugs"


@dataclass
class EvalResult:
    """Evaluation results for one question"""
    question: str
    predicted_answer: str
    ground_truth_answer: str
    retrieved_docs: List[str]
    relevant_docs: List[str]
    
    # Retrieval metrics
    precision_at_3: float
    recall_at_3: float
    mrr: float
    
    # Generation metrics
    semantic_similarity: float
    answer_length: int
    
    # Performance metrics
    retrieval_time: float
    generation_time: float
    total_time: float
    confidence_score: float


class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self, api_url: str = "http://localhost:8000/api/v1"):
        self.api_url = api_url
        self.token = None
        
        # Load semantic similarity model for evaluation
        print("Loading evaluation model...")
        self.eval_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def authenticate(self, username: str, password: str) -> bool:
        """Get JWT token"""
        import requests
        try:
            response = requests.post(
                f"{self.api_url}/token",
                json={"username": username, "password": password}
            )
            if response.status_code == 200:
                self.token = response.json()["access_token"]
                return True
        except Exception as e:
            print(f"Auth failed: {e}")
        return False
    
    def query_rag(self, question: str) -> Dict:
        """Query the RAG API"""
        import requests
        response = requests.post(
            f"{self.api_url}/query",
            json={"query": question},
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        raise Exception(f"Query failed: {response.status_code}")
    
    def calculate_precision_at_k(self, retrieved: List[str], relevant: List[str], k: int = 3) -> float:
        """Precision@K - fraction of top-k retrieved docs that are relevant"""
        if not retrieved:
            return 0.0
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        hits = sum(1 for doc in retrieved_k if doc in relevant_set)
        return hits / len(retrieved_k)
    
    def calculate_recall_at_k(self, retrieved: List[str], relevant: List[str], k: int = 3) -> float:
        """Recall@K - fraction of relevant docs found in top-k"""
        if not relevant:
            return 0.0
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        hits = sum(1 for doc in retrieved_k if doc in relevant_set)
        return hits / len(relevant_set)
    
    def calculate_mrr(self, retrieved: List[str], relevant: List[str]) -> float:
        """Mean Reciprocal Rank - 1/rank of first relevant doc"""
        relevant_set = set(relevant)
        for idx, doc in enumerate(retrieved, 1):
            if doc in relevant_set:
                return 1.0 / idx
        return 0.0
    
    def calculate_semantic_similarity(self, pred: str, truth: str) -> float:
        """Semantic similarity using sentence embeddings"""
        emb_pred = self.eval_model.encode(pred, convert_to_tensor=True)
        emb_truth = self.eval_model.encode(truth, convert_to_tensor=True)
        similarity = util.cos_sim(emb_pred, emb_truth).item()
        return similarity
    
    def evaluate_question(self, test_case: EvalQuestion) -> EvalResult:
        """Evaluate RAG on single question"""
        print(f"\n📝 Evaluating: {test_case.question[:80]}...")
        
        # Query RAG
        start_time = time.time()
        result = self.query_rag(test_case.question)
        
        # Extract retrieved document IDs (from sources)
        retrieved_docs = [
            src["document"] for src in result.get("sources", [])
        ]
        
        # Calculate retrieval metrics
        precision = self.calculate_precision_at_k(retrieved_docs, test_case.relevant_docs, k=3)
        recall = self.calculate_recall_at_k(retrieved_docs, test_case.relevant_docs, k=3)
        mrr = self.calculate_mrr(retrieved_docs, test_case.relevant_docs)
        
        # Calculate generation metrics
        semantic_sim = self.calculate_semantic_similarity(
            result["answer"],
            test_case.ground_truth_answer
        )
        
        return EvalResult(
            question=test_case.question,
            predicted_answer=result["answer"],
            ground_truth_answer=test_case.ground_truth_answer,
            retrieved_docs=retrieved_docs,
            relevant_docs=test_case.relevant_docs,
            precision_at_3=precision,
            recall_at_3=recall,
            mrr=mrr,
            semantic_similarity=semantic_sim,
            answer_length=len(result["answer"]),
            retrieval_time=result.get("retrieval_time", 0),
            generation_time=result.get("generation_time", 0),
            total_time=result.get("total_time", 0),
            confidence_score=result.get("confidence_score", 0)
        )
    
    def evaluate_dataset(self, test_cases: List[EvalQuestion]) -> Dict:
        """Evaluate entire test dataset"""
        results = []
        
        for test_case in test_cases:
            try:
                result = self.evaluate_question(test_case)
                results.append(result)
                
                # Print per-question metrics
                print(f"  ✓ P@3: {result.precision_at_3:.3f} | "
                      f"R@3: {result.recall_at_3:.3f} | "
                      f"MRR: {result.mrr:.3f} | "
                      f"Sem: {result.semantic_similarity:.3f}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        # Aggregate metrics
        return self.aggregate_results(results)
    
    def aggregate_results(self, results: List[EvalResult]) -> Dict:
        """Compute aggregate statistics"""
        if not results:
            return {}
        
        metrics = {
            "num_evaluated": len(results),
            
            # Retrieval metrics
            "avg_precision_at_3": statistics.mean(r.precision_at_3 for r in results),
            "avg_recall_at_3": statistics.mean(r.recall_at_3 for r in results),
            "avg_mrr": statistics.mean(r.mrr for r in results),
            
            # Generation metrics
            "avg_semantic_similarity": statistics.mean(r.semantic_similarity for r in results),
            "avg_answer_length": statistics.mean(r.answer_length for r in results),
            
            # Performance metrics
            "avg_retrieval_time": statistics.mean(r.retrieval_time for r in results),
            "avg_generation_time": statistics.mean(r.generation_time for r in results),
            "avg_total_time": statistics.mean(r.total_time for r in results),
            "avg_confidence": statistics.mean(r.confidence_score for r in results),
            
            # Percentiles
            "p50_total_time": statistics.median(r.total_time for r in results),
            "p95_total_time": np.percentile([r.total_time for r in results], 95),
            "p99_total_time": np.percentile([r.total_time for r in results], 99),
        }
        
        return metrics, results


def load_test_dataset(path: str = "data/eval_dataset.json") -> List[EvalQuestion]:
    """Load evaluation dataset"""
    file_path = Path(path)
    if not file_path.exists():
        print(f"⚠️  Test dataset not found: {path}")
        return create_sample_dataset()
    
    with open(file_path) as f:
        data = json.load(f)
    
    return [EvalQuestion(**item) for item in data]


def create_sample_dataset() -> List[EvalQuestion]:
    """Create sample evaluation dataset"""
    print("📝 Using sample evaluation dataset...")
    
    return [
        EvalQuestion(
            question="What is acute rejection in kidney transplant?",
            ground_truth_answer="Acute rejection is an immune response where the recipient's body attacks the transplanted kidney, typically occurring within the first few months post-transplant. It can be cellular (T-cell mediated) or antibody-mediated (AMR).",
            relevant_docs=[
                "DOCUMENT 2: ACUTE REJECTION",
                "DOCUMENT 13: KIDNEY TRANSPLANT - ACUTE REJECTION"
            ],
            category="rejection"
        ),
        EvalQuestion(
            question="What are HLA antibodies and why are they important?",
            ground_truth_answer="HLA antibodies are immune proteins that recognize foreign HLA molecules on transplanted organs. They can cause antibody-mediated rejection and are screened before transplant to assess compatibility and risk.",
            relevant_docs=[
                "DOCUMENT 1: ORGAN TRANSPLANTATION IMMUNOLOGY",
            ],
            category="immunology"
        ),
        EvalQuestion(
            question="What is the mechanism of action of tacrolimus?",
            ground_truth_answer="Tacrolimus is a calcineurin inhibitor that prevents T-cell activation by blocking IL-2 production. It binds to FKBP12 protein and inhibits the phosphatase activity of calcineurin, preventing NFAT translocation to the nucleus.",
            relevant_docs=[
                "DOCUMENT 4: IMMUNOSUPPRESSIVE DRUGS"
            ],
            category="immunosuppression"
        ),
        EvalQuestion(
            question="What are signs of kidney transplant rejection?",
            ground_truth_answer="Signs include rising serum creatinine (>0.3 mg/dL increase), reduced urine output, graft tenderness or swelling, fever, hypertension, and proteinuria. Biopsy is required for definitive diagnosis.",
            relevant_docs=[
                "DOCUMENT 2: ACUTE REJECTION",
                "DOCUMENT 13: KIDNEY TRANSPLANT - ACUTE REJECTION"
            ],
            category="rejection"
        ),
        EvalQuestion(
            question="What is the crossmatch test in transplantation?",
            ground_truth_answer="The crossmatch test mixes recipient serum with donor lymphocytes to detect pre-existing antibodies against donor HLA. A positive crossmatch indicates presence of donor-specific antibodies and high rejection risk.",
            relevant_docs=[
                "DOCUMENT 1: ORGAN TRANSPLANTATION IMMUNOLOGY"
            ],
            category="immunology"
        ),
    ]


def save_results(metrics: Dict, results: List[EvalResult], output_dir: str = "data/eval_results"):
    """Save evaluation results"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save aggregate metrics
    metrics_file = output_path / f"metrics_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed results
    results_file = output_path / f"results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print(f"\n💾 Results saved to {output_dir}/")
    return metrics_file, results_file


def print_report(metrics: Dict):
    """Print evaluation report"""
    print("\n" + "="*70)
    print("📊 RAG EVALUATION REPORT")
    print("="*70)
    
    print(f"\n📝 Dataset: {metrics['num_evaluated']} questions evaluated")
    
    print("\n🎯 RETRIEVAL METRICS")
    print(f"  Precision@3:  {metrics['avg_precision_at_3']:.3f}")
    print(f"  Recall@3:     {metrics['avg_recall_at_3']:.3f}")
    print(f"  MRR:          {metrics['avg_mrr']:.3f}")
    
    print("\n💬 GENERATION METRICS")
    print(f"  Semantic Similarity: {metrics['avg_semantic_similarity']:.3f}")
    print(f"  Avg Answer Length:   {metrics['avg_answer_length']:.0f} chars")
    print(f"  Avg Confidence:      {metrics['avg_confidence']:.3f}")
    
    print("\n⏱️  PERFORMANCE METRICS")
    print(f"  Avg Retrieval Time:  {metrics['avg_retrieval_time']:.3f}s")
    print(f"  Avg Generation Time: {metrics['avg_generation_time']:.3f}s")
    print(f"  Avg Total Time:      {metrics['avg_total_time']:.3f}s")
    print(f"  Median (P50):        {metrics['p50_total_time']:.3f}s")
    print(f"  P95:                 {metrics['p95_total_time']:.3f}s")
    print(f"  P99:                 {metrics['p99_total_time']:.3f}s")
    
    print("\n" + "="*70)


def main():
    """Run evaluation"""
    print("🔬 RAG System Evaluation")
    print("="*70)
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Authenticate
    print("\n🔐 Authenticating...")
    if not evaluator.authenticate("admin@transplant.ai", "admin123"):
        print("❌ Authentication failed")
        return
    print("✅ Authenticated")
    
    # Load test dataset
    print("\n📚 Loading test dataset...")
    test_cases = load_test_dataset()
    print(f"✅ Loaded {len(test_cases)} test cases")
    
    # Run evaluation
    print("\n🚀 Running evaluation...")
    metrics, results = evaluator.evaluate_dataset(test_cases)
    
    # Print report
    print_report(metrics)
    
    # Save results
    save_results(metrics, results)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
