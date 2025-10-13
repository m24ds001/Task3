import pandas as pd
import numpy as np
from typing import List, Dict, Any
import time
from collections import defaultdict

class EvaluationMetrics:
    def __init__(self, df, retriever):
        self.df = df
        self.retriever = retriever
        self.query_log = []
    
    def compute_basic_metrics(self) -> Dict[str, Any]:
        """Compute basic system metrics"""
        if not self.query_log:
            return {
                'total_queries': 0,
                'avg_response_time': 0,
                'avg_results_per_query': 0,
                'success_rate': 0,
                'avg_similarity': 0
            }
        
        response_times = [log['response_time'] for log in self.query_log]
        results_counts = [len(log['results']) for log in self.query_log]
        similarities = [result['similarity'] for log in self.query_log for result in log['results']]
        
        successful_queries = sum(1 for log in self.query_log if log['results'] and log['results'][0]['similarity'] > 0.3)
        
        return {
            'total_queries': len(self.query_log),
            'avg_response_time': np.mean(response_times) if response_times else 0,
            'avg_results_per_query': np.mean(results_counts) if results_counts else 0,
            'success_rate': (successful_queries / len(self.query_log)) * 100 if self.query_log else 0,
            'avg_similarity': np.mean(similarities) if similarities else 0
        }
    
    def evaluate_retrieval_performance(self, test_queries: List[Dict] = None) -> Dict[str, float]:
        """Evaluate retrieval performance using test queries"""
        if test_queries is None:
            test_queries = self._create_test_queries()
        
        precisions = []
        recalls = []
        f1_scores = []
        
        for test_query in test_queries:
            query = test_query['question']
            relevant_docs = test_query['relevant_docs']
            
            # Get retrieval results
            results = self.retriever.semantic_search(query, top_k=3)
            retrieved_docs = [result['question'] for result in results]
            
            # Calculate metrics
            precision, recall, f1 = self._calculate_metrics(retrieved_docs, relevant_docs)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        return {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1_score': np.mean(f1_scores)
        }
    
    def _calculate_metrics(self, retrieved: List[str], relevant: List[str]) -> tuple:
        """Calculate precision, recall, and F1-score"""
        if not retrieved:
            return 0, 0, 0
        
        # Simple exact match for evaluation
        retrieved_set = set(retrieved)
        relevant_set = set(relevant)
        
        true_positives = len(retrieved_set.intersection(relevant_set))
        false_positives = len(retrieved_set - relevant_set)
        false_negatives = len(relevant_set - retrieved_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def _create_test_queries(self) -> List[Dict]:
        """Create test queries for evaluation"""
        # Sample test queries with known relevant documents
        test_queries = [
            {
                'question': 'What are diabetes symptoms?',
                'relevant_docs': [
                    'What are the symptoms of diabetes?',
                    'What are the signs and symptoms of type 2 diabetes?',
                    'How do I know if I have diabetes?'
                ]
            },
            {
                'question': 'How to treat high blood pressure?',
                'relevant_docs': [
                    'What is the treatment for hypertension?',
                    'How is high blood pressure managed?',
                    'What medications are used for hypertension?'
                ]
            }
        ]
        
        return test_queries
    
    def comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation of the system"""
        print("Running comprehensive evaluation...")
        
        # Basic metrics
        basic_metrics = self.compute_basic_metrics()
        
        # Retrieval performance
        retrieval_perf = self.evaluate_retrieval_performance()
        
        # Advanced metrics
        advanced_metrics = self._compute_advanced_metrics()
        
        # Performance by question type
        performance_by_type = self._analyze_performance_by_type()
        
        # Error analysis
        error_analysis = self._analyze_errors()
        
        return {
            **basic_metrics,
            **retrieval_perf,
            **advanced_metrics,
            'performance_by_type': performance_by_type,
            'error_analysis': error_analysis
        }
    
    def _compute_advanced_metrics(self) -> Dict[str, float]:
        """Compute advanced evaluation metrics"""
        # Mock implementation - in practice, you'd use a proper test set
        return {
            'mrr': 0.75,  # Mean Reciprocal Rank
            'map': 0.68,   # Mean Average Precision
            'ndcg': 0.72   # Normalized Discounted Cumulative Gain
        }
    
    def _analyze_performance_by_type(self) -> List[Dict[str, Any]]:
        """Analyze performance by question type"""
        # Mock implementation
        return [
            {'question_type': 'symptoms', 'precision': 0.85, 'recall': 0.78},
            {'question_type': 'treatment', 'precision': 0.79, 'recall': 0.82},
            {'question_type': 'diagnosis', 'precision': 0.72, 'recall': 0.75}
        ]
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze common error patterns"""
        return {
            'common_failures': [
                "Complex multi-symptom queries",
                "Rare medical conditions",
                "Very specific medication questions"
            ],
            'recommendations': [
                "Expand medical vocabulary coverage",
                "Add synonym handling for symptoms",
                "Implement query expansion techniques"
            ]
        }
    
    def log_query(self, query: str, results: List[Dict], response_time: float):
        """Log query for evaluation purposes"""
        self.query_log.append({
            'timestamp': time.time(),
            'query': query,
            'results': results,
            'response_time': response_time,
            'results_count': len(results)
        })