#!/usr/bin/env python3
"""
Research-grade model comparison for medical RAG systems
Evaluates compact, open-weight LLMs deployable on consumer-grade GPUs (≤4 GB VRAM)
"""
import requests
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import re

API_URL = "http://localhost:8000/api/v1"

# Authenticate
response = requests.post(f"{API_URL}/token", json={"username": "admin@transplant.ai", "password": "admin123"})
token = response.json()["access_token"]

# Test questions for comprehensive evaluation (reasoning-heavy for model differentiation)
test_questions = [
    "Differentiate acute cellular rejection from antibody-mediated rejection using biopsy and laboratory findings.",
    "Explain the mechanism of tacrolimus nephrotoxicity and how it is distinguished from rejection.",
    "When is virtual crossmatch unsafe despite negative donor-specific antibodies?",
    "Compare induction therapy strategies for high-risk vs standard-risk transplant recipients.",
    "Describe the diagnostic criteria and treatment approach for chronic antibody-mediated rejection."
]

# Research-grade model selection (4 models for paper)
# Principle: Compact models deployable on ≤4 GB VRAM for privacy-preserving medical RAG
models = [
    "gemma3:1b",      # Extreme low-resource baseline (smallest)
    "gemma2:2b",      # Efficiency-first (best latency/quality)
    "llama3.2:3b",    # General LLM baseline (widely cited)
    "phi3:mini",      # Medical-oriented (strong reasoning)
]

model_labels = {
    "gemma3:1b": "Gemma 3 (1B)",
    "gemma2:2b": "Gemma 2 (2B)",
    "llama3.2:3b": "LLaMA 3.2 (3B)",
    "phi3:mini": "Phi-3 Mini (3.8B)",
}

model_categories = {
    "gemma3:1b": "Ultra-compact",
    "gemma2:2b": "Efficiency-first",
    "llama3.2:3b": "General baseline",
    "phi3:mini": "Medical-oriented",
}

print("\n" + "="*80)
print("🔬 RESEARCH-GRADE MODEL COMPARISON FOR MEDICAL RAG")
print("="*80)
print("\n📋 Evaluation Framework:")
print("   • Compact models deployable on consumer GPUs (≤4 GB VRAM)")
print("   • Privacy-preserving local inference")
print(f"   • {len(test_questions)} test queries across medical domains")
print(f"   • {len(models)} model configurations")
print("\n" + "="*80)

results = {}

for model in models:
    print(f"\n🧪 Testing: {model_labels[model]} ({model})")
    print(f"   Category: {model_categories[model]}")
    print("   " + "-"*70)
    
    model_results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n   Query {i}/{len(test_questions)}: {question[:60]}...")
        
        start = time.time()
        response = requests.post(
            f"{API_URL}/query",
            json={
                "query": question,
                "model": model,
                "temperature": 0.2,
                "max_tokens": 400,
                "top_p": 0.9,
                "confidence_threshold": 0.0  # CRITICAL: Bypass confidence gating for research eval
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=180
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            
            # ======================================================================
            # SCIENTIFIC FAITHFULNESS METRIC (Research-grade)
            # ======================================================================
            answer_text = result['answer']
            sources = result.get('sources', [])
            
            # Extract context from sources
            context_chunks = [chunk.get('text_preview', '') for chunk in sources]
            full_context = ' '.join(context_chunks)
            
            # Method: Token-level overlap with medical term weighting
            def calculate_faithfulness(answer: str, context: str) -> dict:
                """
                Calculate faithfulness using multiple metrics:
                1. Token overlap (content words only)
                2. Medical term coverage
                3. Citation density
                """
                if not answer or not context:
                    return {"score": 0.0, "method": "empty", "details": "Empty answer or context"}
                
                # Normalize texts
                answer_lower = answer.lower()
                context_lower = context.lower()
                
                # Extract content words (remove stop words)
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                             'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                             'being', 'this', 'that', 'these', 'those', 'it', 'its', 'can', 'will',
                             'may', 'also', 'such', 'which', 'their', 'has', 'have', 'had'}
                
                # Tokenize
                answer_tokens = [w for w in re.findall(r'\b\w+\b', answer_lower) if len(w) > 2 and w not in stop_words]
                context_tokens = set(re.findall(r'\b\w+\b', context_lower))
                
                if not answer_tokens:
                    return {"score": 0.0, "method": "no_content", "details": "No content tokens"}
                
                # Calculate overlap
                matched_tokens = [t for t in answer_tokens if t in context_tokens]
                token_overlap = len(matched_tokens) / len(answer_tokens)
                
                # Medical term boost (domain-specific vocabulary)
                medical_terms = ['rejection', 'antibody', 'transplant', 'tacrolimus', 'immunosuppression',
                                'nephrotoxicity', 'biopsy', 'creatinine', 'donor', 'recipient', 'allograft',
                                'crossmatch', 'hla', 'lymphocyte', 'cytokine', 'protocol', 'therapy',
                                'diagnosis', 'treatment', 'clinical', 'acute', 'chronic', 'graft']
                
                medical_in_answer = sum(1 for term in medical_terms if term in answer_lower)
                medical_in_context = sum(1 for term in medical_terms if term in context_lower and term in answer_lower)
                medical_coverage = medical_in_context / medical_in_answer if medical_in_answer > 0 else 0.5
                
                # Combined score (weighted average)
                faithfulness_score = (0.7 * token_overlap) + (0.3 * medical_coverage)
                faithfulness_score = min(1.0, faithfulness_score)  # Cap at 1.0
                
                return {
                    "score": faithfulness_score,
                    "method": "token_overlap_weighted",
                    "details": {
                        "token_overlap": round(token_overlap, 3),
                        "medical_coverage": round(medical_coverage, 3),
                        "answer_tokens": len(answer_tokens),
                        "matched_tokens": len(matched_tokens),
                        "medical_terms_used": medical_in_answer
                    }
                }
            
            faith_result = calculate_faithfulness(answer_text, full_context)
            faithfulness = faith_result["score"]
            hallucination_rate = max(0, 1 - faithfulness)
            
            model_results.append({
                "question": question,
                "total_time": elapsed,
                "retrieval_time": result['retrieval_time'],
                "generation_time": result['generation_time'],
                "total_tokens": result['total_tokens'],
                "confidence_score": result['confidence_score'],
                "answer_length": len(result['answer']),
                "tokens_per_second": result['total_tokens'] / result['generation_time'] if result['generation_time'] > 0 else 0,
                "faithfulness": faithfulness,
                "faithfulness_details": faith_result["details"],
                "hallucination_rate": hallucination_rate,
                "answer": result['answer']
            })
            
            print(f"      ✅ {elapsed:.2f}s | Gen: {result['generation_time']:.2f}s | {result['total_tokens']} tok | "
                  f"Faith: {faithfulness:.3f} | Len: {len(result['answer'])}")
        else:
            print(f"      ❌ Failed: {response.status_code}")
            model_results.append(None)
    
    # Calculate aggregate metrics
    valid_results = [r for r in model_results if r is not None]
    if valid_results:
        results[model] = {
            "model_label": model_labels[model],
            "category": model_categories[model],
            "queries": model_results,
            "avg_total_time": np.mean([r['total_time'] for r in valid_results]),
            "avg_retrieval_time": np.mean([r['retrieval_time'] for r in valid_results]),
            "avg_generation_time": np.mean([r['generation_time'] for r in valid_results]),
            "avg_tokens": np.mean([r['total_tokens'] for r in valid_results]),
            "avg_confidence": np.mean([r['confidence_score'] for r in valid_results]),
            "avg_answer_length": np.mean([r['answer_length'] for r in valid_results]),
            "avg_tokens_per_sec": np.mean([r['tokens_per_second'] for r in valid_results]),
            "avg_faithfulness": np.mean([r['faithfulness'] for r in valid_results]),
            "avg_hallucination": np.mean([r['hallucination_rate'] for r in valid_results]),
            "p95_latency": np.percentile([r['total_time'] for r in valid_results], 95),
            "success_rate": len(valid_results) / len(test_questions)
        }
        
        print(f"\n   📊 Summary Statistics:")
        print(f"      Avg Total Time: {results[model]['avg_total_time']:.2f}s")
        print(f"      Avg Gen Time: {results[model]['avg_generation_time']:.2f}s")
        print(f"      P95 Latency: {results[model]['p95_latency']:.2f}s")
        print(f"      Avg Throughput: {results[model]['avg_tokens_per_sec']:.1f} tokens/sec")
        print(f"      Avg Faithfulness: {results[model]['avg_faithfulness']:.3f}")
        print(f"      Avg Hallucination: {results[model]['avg_hallucination']:.3f}")
        print(f"      Success Rate: {results[model]['success_rate']*100:.0f}%")

print("\n" + "="*80)

# Save results
output_dir = Path("data/eval_results")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = output_dir / f"research_model_comparison_{timestamp}.json"

with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {results_file}")

# Generate research-quality publication graphs
if results:
    print("\n📊 Generating publication-quality visualizations...")
    
    # Set academic publication style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8
    })
    
    model_list = [m for m in models if m in results]
    model_names = [model_labels[m] for m in model_list]
    categories = [model_categories[m] for m in model_list]
    
    # Color scheme: gradient from smallest to largest models
    colors = ['#9b59b6', '#3498db', '#e74c3c', '#f39c12'][:len(model_list)]
    
    # ============================================================================
    # Figure 1: Comprehensive 6-Panel Comparison
    # ============================================================================
    fig = plt.figure(figsize=(16, 10))
    
    # Panel 1: Average Response Time
    ax1 = plt.subplot(2, 3, 1)
    avg_times = [results[m]['avg_total_time'] for m in model_list]
    bars1 = ax1.bar(range(len(model_names)), avg_times, color=colors, alpha=0.85, 
                    edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Time (seconds)', fontweight='bold', fontsize=11)
    ax1.set_title('(A) Average Response Time', fontweight='bold', loc='left', fontsize=12)
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=25, ha='right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars1, avg_times)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + max(avg_times)*0.02, 
                f'{val:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Panel 2: P95 Latency (Critical for Production)
    ax2 = plt.subplot(2, 3, 2)
    p95_latencies = [results[m]['p95_latency'] for m in model_list]
    bars2 = ax2.bar(range(len(model_names)), p95_latencies, color=colors, alpha=0.85, 
                    edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Time (seconds)', fontweight='bold', fontsize=11)
    ax2.set_title('(B) P95 Latency', fontweight='bold', loc='left', fontsize=12)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=25, ha='right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars2, p95_latencies)):
        ax2.text(bar.get_x() + bar.get_width()/2, val + max(p95_latencies)*0.02, 
                f'{val:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Panel 3: Generation Throughput
    ax3 = plt.subplot(2, 3, 3)
    throughputs = [results[m]['avg_tokens_per_sec'] for m in model_list]
    bars3 = ax3.bar(range(len(model_names)), throughputs, color=colors, alpha=0.85, 
                    edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Tokens/Second', fontweight='bold', fontsize=11)
    ax3.set_title('(C) Generation Throughput', fontweight='bold', loc='left', fontsize=12)
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels(model_names, rotation=25, ha='right', fontsize=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars3, throughputs)):
        ax3.text(bar.get_x() + bar.get_width()/2, val + max(throughputs)*0.02, 
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Panel 4: Stacked Time Breakdown
    ax4 = plt.subplot(2, 3, 4)
    retrieval_times = [results[m]['avg_retrieval_time'] for m in model_list]
    generation_times = [results[m]['avg_generation_time'] for m in model_list]
    x_pos = np.arange(len(model_names))
    
    bars4a = ax4.bar(x_pos, retrieval_times, label='Retrieval', 
                     color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.2)
    bars4b = ax4.bar(x_pos, generation_times, bottom=retrieval_times, 
                     label='Generation', color='#e74c3c', alpha=0.85, 
                     edgecolor='black', linewidth=1.2)
    ax4.set_ylabel('Time (seconds)', fontweight='bold', fontsize=11)
    ax4.set_title('(D) Time Breakdown (Stacked)', fontweight='bold', loc='left', fontsize=12)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(model_names, rotation=25, ha='right', fontsize=10)
    ax4.legend(loc='upper right', frameon=True, shadow=True, fontsize=9)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Panel 5: Retrieval Confidence (Model-Independent)
    ax5 = plt.subplot(2, 3, 5)
    confidences = [results[m]['avg_confidence'] for m in model_list]
    bars5 = ax5.bar(range(len(model_names)), confidences, color=colors, alpha=0.85, 
                    edgecolor='black', linewidth=1.2)
    ax5.set_ylabel('Retrieval Confidence', fontweight='bold', fontsize=11)
    ax5.set_title('(E) Retrieval Confidence*', fontweight='bold', loc='left', fontsize=12)
    ax5.set_xticks(range(len(model_names)))
    ax5.set_xticklabels(model_names, rotation=25, ha='right', fontsize=10)
    ax5.set_ylim(0, 1.0)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    ax5.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
    ax5.text(0.5, -0.25, '*Identical across models (shared retriever)', 
            transform=ax5.transAxes, ha='center', fontsize=7, style='italic')
    for i, (bar, val) in enumerate(zip(bars5, confidences)):
        ax5.text(bar.get_x() + bar.get_width()/2, val + 0.03, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Panel 6: Answer Completeness
    ax6 = plt.subplot(2, 3, 6)
    answer_lengths = [results[m]['avg_answer_length'] for m in model_list]
    bars6 = ax6.bar(range(len(model_names)), answer_lengths, color=colors, alpha=0.85, 
                    edgecolor='black', linewidth=1.2)
    ax6.set_ylabel('Character Count', fontweight='bold', fontsize=11)
    ax6.set_title('(F) Answer Completeness', fontweight='bold', loc='left', fontsize=12)
    ax6.set_xticks(range(len(model_names)))
    ax6.set_xticklabels(model_names, rotation=25, ha='right', fontsize=10)
    ax6.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars6, answer_lengths)):
        ax6.text(bar.get_x() + bar.get_width()/2, val + max(answer_lengths)*0.02, 
                f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    plt.suptitle('Model Performance Comparison for Privacy-Preserving Medical RAG', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    fig_file = output_dir / f"research_comparison_detailed_{timestamp}.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Detailed comparison: {fig_file}")
    plt.close(fig)
    
    # Create completely separate, standalone individual graphs (not cutouts)
    individual_dir = output_dir / f"individual_graphs_{timestamp}"
    individual_dir.mkdir(exist_ok=True)
    
    # Helper function to create clean standalone graphs
    def create_standalone_graph(data, title, ylabel, filename, value_format=".2f", ylim=None):
        fig_single, ax_single = plt.subplots(figsize=(10, 7))
        bars = ax_single.bar(model_names, data, color=colors, alpha=0.85, 
                            edgecolor='black', linewidth=1.5, width=0.6)
        
        # Add value labels on bars
        for bar, val in zip(bars, data):
            height = bar.get_height()
            ax_single.text(bar.get_x() + bar.get_width()/2., height,
                          f'{val:{value_format}}',
                          ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        ax_single.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax_single.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax_single.tick_params(axis='y', labelsize=12)
        ax_single.tick_params(axis='x', labelsize=13)
        ax_single.set_xticklabels(model_names, rotation=0, ha='center', fontweight='normal')
        ax_single.grid(axis='y', alpha=0.3, linestyle='--')
        ax_single.set_axisbelow(True)
        if ylim:
            ax_single.set_ylim(ylim)
        
        plt.tight_layout()
        plt.savefig(individual_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig_single)
    
    # Generate each individual graph as completely separate figure
    avg_times = [results[m]['avg_total_time'] for m in model_list]
    p95_latencies = [results[m]['p95_latency'] for m in model_list]
    throughputs = [results[m]['avg_tokens_per_sec'] for m in model_list]
    avg_gen_times = [results[m]['avg_generation_time'] for m in model_list]
    confidences = [results[m]['avg_confidence'] for m in model_list]
    answer_lengths = [results[m]['avg_answer_length'] for m in model_list]
    
    create_standalone_graph(avg_times, 'Average Response Time', 'Time (seconds)', 
                           'fig_0_average_response_time.png')
    create_standalone_graph(p95_latencies, 'P95 Latency', 'Time (seconds)', 
                           'fig_1_p95_latency.png')
    create_standalone_graph(throughputs, 'Generation Throughput', 'Tokens/second', 
                           'fig_2_generation_throughput.png', '.1f')
    create_standalone_graph(answer_lengths, 'Answer Completeness', 'Character Count', 
                           'fig_5_answer_completeness.png', '.0f')
    create_standalone_graph(confidences, 'Retrieval Confidence*\n*Model-independent (shared retriever)', 
                           'Confidence Score', 'fig_4_answer_confidence.png', '.3f', (0, 1.0))
    
    # Time breakdown (stacked bar) - special handling
    fig_single, ax_single = plt.subplots(figsize=(10, 7))
    retrieval_times = [results[m]['avg_retrieval_time'] for m in model_list]
    bars1 = ax_single.bar(model_names, retrieval_times, label='Retrieval', 
                         color='#95a5a6', alpha=0.85, edgecolor='black', linewidth=1.5, width=0.6)
    bars2 = ax_single.bar(model_names, avg_gen_times, bottom=retrieval_times, 
                         label='Generation', color=colors, alpha=0.85, edgecolor='black', linewidth=1.5, width=0.6)
    
    # Add total time labels
    for i, (ret, gen) in enumerate(zip(retrieval_times, avg_gen_times)):
        total = ret + gen
        ax_single.text(i, total, f'{total:.2f}s', ha='center', va='bottom', 
                      fontsize=13, fontweight='bold')
    
    ax_single.set_ylabel('Time (seconds)', fontsize=14, fontweight='bold')
    ax_single.set_title('Time Breakdown: Retrieval vs Generation', fontsize=16, fontweight='bold', pad=20)
    ax_single.legend(fontsize=13, loc='upper left', frameon=True, shadow=True)
    ax_single.tick_params(axis='y', labelsize=12)
    ax_single.tick_params(axis='x', labelsize=13)
    ax_single.set_xticklabels(model_names, rotation=0, ha='center', fontweight='normal')
    ax_single.grid(axis='y', alpha=0.3, linestyle='--')
    ax_single.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(individual_dir / 'fig_3_time_breakdown.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_single)
    
    print(f"   ✅ Individual graphs: {individual_dir}")
    
    # ============================================================================
    # Figure 2: Normalized Performance Radar/Summary
    # ============================================================================
    fig2, ax = plt.subplots(figsize=(14, 8))
    
    # Normalize metrics (0-1 scale, higher is better)
    max_throughput = max(throughputs) if max(throughputs) > 0 else 1
    max_time = max(avg_times)
    max_answer_len = max(answer_lengths) if max(answer_lengths) > 0 else 1
    
    metrics = {
        'Speed\n(1/latency)': [max_time/results[m]['avg_total_time'] for m in model_list],
        'Throughput\n(tok/s)': [results[m]['avg_tokens_per_sec']/max_throughput for m in model_list],
        'Confidence': [results[m]['avg_confidence'] for m in model_list],
        'Completeness': [results[m]['avg_answer_length']/max_answer_len for m in model_list],
    }
    
    x_pos = np.arange(len(metrics))
    width = 0.18
    
    for i, (model, color) in enumerate(zip(model_list, colors)):
        values = [metrics[metric][i] for metric in metrics]
        offset = width * (i - len(model_list)/2 + 0.5)
        bars = ax.bar(x_pos + offset, values, width, label=model_labels[model], 
                     color=color, alpha=0.85, edgecolor='black', linewidth=1.2)
        
        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Normalized Score (Higher is Better)', fontweight='bold', fontsize=12)
    ax.set_title('Normalized Performance Comparison Across Key Metrics', 
                fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics.keys(), fontweight='bold', fontsize=11)
    ax.legend(frameon=True, shadow=True, loc='upper right', ncol=2)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    
    plt.tight_layout()
    
    fig2_file = output_dir / f"research_comparison_normalized_{timestamp}.png"
    plt.savefig(fig2_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Normalized comparison: {fig2_file}")
    
    # ============================================================================
    # Figure 3: Research Table (as text figure for LaTeX inclusion)
    # ============================================================================
    print("\n📋 RESEARCH PAPER TABLE (LaTeX-ready):")
    print("-" * 120)
    print(f"{'Model':<20} {'Category':<18} {'Gen Time':<12} {'P95':<10} {'Tok/s':<10} {'Faith':<10} {'Halluc ↓':<10} {'Length':<8}")
    print("-" * 120)
    for model in model_list:
        print(f"{model_labels[model]:<20} "
              f"{model_categories[model]:<18} "
              f"{results[model]['avg_generation_time']:<12.2f} "
              f"{results[model]['p95_latency']:<10.2f} "
              f"{results[model]['avg_tokens_per_sec']:<10.1f} "
              f"{results[model]['avg_faithfulness']:<10.3f} "
              f"{results[model]['avg_hallucination']:<10.3f} "
              f"{int(results[model]['avg_answer_length']):<8}")
    print("-" * 120)
    
    # Publishable conclusion
    best_speed = min(model_list, key=lambda m: results[m]['avg_generation_time'])
    best_quality = max(model_list, key=lambda m: results[m]['avg_faithfulness'])
    best_balanced = min(model_list, key=lambda m: results[m]['avg_generation_time'] / (results[m]['avg_faithfulness'] + 0.01))
    
    print("\n🏆 RESEARCH CONCLUSIONS:")
    print(f"   Fastest: {model_labels[best_speed]} ({results[best_speed]['avg_generation_time']:.2f}s)")
    print(f"   Highest Quality: {model_labels[best_quality]} (faithfulness: {results[best_quality]['avg_faithfulness']:.3f})")
    print(f"   Best Balance: {model_labels[best_balanced]}")
    print(f"\n💡 Paper Claim: '{model_labels[best_balanced]} achieved the best trade-off between")
    print(f"   factual faithfulness and inference latency for privacy-preserving medical RAG.'")
    
    print("\n✨ All visualizations generated successfully!")
    print(f"   📁 Location: {output_dir.absolute()}")
    print(f"\n💡 For your paper: Use Figure 1 for comprehensive analysis,")
    print(f"   Figure 2 for at-a-glance comparison in abstract/intro.")
else:
    print("\n⚠️  No results to visualize")
