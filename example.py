#!/usr/bin/env python3
"""
Example script demonstrating Intent Mining using ITER-DBSCAN
"""

import pandas as pd
import numpy as np
from ITER_DBSCAN import ITER_DBSCAN
from sentenceEmbedding import SentenceEmbedding
from evaluation import EvaluateDataset


def create_sample_data():
    """
    Create sample conversation data for demonstration
    """
    sample_texts = [
        "How do I reset my password?",
        "I forgot my password, can you help?",
        "Reset password please",
        "I need to change my password",
        "Can you help me change my password?",
        "How do I delete my account?",
        "I want to remove my account",
        "Delete my profile please",
        "Remove account from system",
        "How do I export my data?",
        "I need to download my information",
        "Export all my data please",
        "Can I get a copy of my data?",
        "How do I sync my accounts?",
        "I want to connect my accounts",
        "Link my profiles together",
        "Synchronize accounts please",
        "How do I filter spam?",
        "I'm getting too many spam messages",
        "Block unwanted emails",
        "Filter out spam please",
        "How do I find alternatives to this app?",
        "Are there better options available?",
        "What are the alternatives?",
        "Can you suggest alternatives?",
        "I want to try something else"
    ]
    
    sample_labels = [
        "Reset Password", "Reset Password", "Reset Password", "Change Password", "Change Password",
        "Delete Account", "Delete Account", "Delete Account", "Delete Account",
        "Export Data", "Export Data", "Export Data", "Export Data",
        "Sync Accounts", "Sync Accounts", "Sync Accounts", "Sync Accounts",
        "Filter Spam", "Filter Spam", "Filter Spam", "Filter Spam",
        "Find Alternatives", "Find Alternatives", "Find Alternatives", "Find Alternatives", "Find Alternatives"
    ]
    
    return sample_texts, sample_labels


def run_example():
    """
    Run a complete example of intent mining
    """
    print("Intent Mining Example using ITER-DBSCAN")
    print("=" * 50)
    
    # Create sample data
    texts, true_labels = create_sample_data()
    print(f"Created sample dataset with {len(texts)} texts and {len(set(true_labels))} intent categories")
    
    # Convert labels to numeric for evaluation
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    numeric_labels = le.fit_transform(true_labels)
    
    # Initialize the embedding model
    print("\nInitializing sentence embedding model...")
    embedding_model = SentenceEmbedding()
    
    # Get embeddings
    print("Computing sentence embeddings...")
    embeddings = embedding_model.getEmbeddings(texts)
    print(f"Generated embeddings with shape: {np.array(embeddings).shape}")
    
    # Run ITER-DBSCAN clustering
    print("\nRunning ITER-DBSCAN clustering...")
    model = ITER_DBSCAN(
        initial_distance=0.3,
        initial_minimum_samples=3,
        delta_distance=0.01,
        delta_minimum_samples=1,
        max_iteration=10,
        threshold=10
    )
    
    cluster_labels = model.fit_predict(texts)
    
    # Evaluate results
    print("\nEvaluating clustering results...")
    evaluator = EvaluateDataset()
    results = evaluator.evaluate_clustering(numeric_labels, cluster_labels, embeddings)
    
    print("\nEvaluation Results:")
    print("-" * 30)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Analyze clusters
    print("\nCluster Analysis:")
    print("-" * 30)
    evaluator.analyze_clusters(texts, cluster_labels, top_n=3)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'text': texts,
        'true_intent': true_labels,
        'cluster_id': cluster_labels
    })
    
    print(f"\nResults saved to 'example_results.csv'")
    results_df.to_csv('example_results.csv', index=False)
    
    # Show cluster distribution
    evaluator.plot_cluster_distribution(cluster_labels, "Example Intent Mining Results")
    
    return results_df, results


if __name__ == "__main__":
    results_df, evaluation_results = run_example()
