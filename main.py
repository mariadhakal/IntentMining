#!/usr/bin/env python3
"""
Main script for Intent Mining using ITER-DBSCAN
This is a modern implementation using sentence-transformers instead of deprecated tensorflow_hub
"""

import pandas as pd
import numpy as np
from ITER_DBSCAN import ITER_DBSCAN
from sentenceEmbedding import SentenceEmbedding
from evaluation import EvaluateDataset
import argparse
import os


def load_data(file_path, text_column='text', label_column=None):
    """
    Load data from various file formats
    
    Args:
        file_path: Path to the data file
        text_column: Name of the column containing text data
        label_column: Name of the column containing true labels (optional)
        
    Returns:
        Tuple of (texts, labels) where labels is None if not provided
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel files.")
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the data. Available columns: {list(df.columns)}")
    
    texts = df[text_column].dropna().tolist()
    labels = None
    
    if label_column and label_column in df.columns:
        labels = df[label_column].dropna().tolist()
        # Ensure texts and labels have same length
        min_len = min(len(texts), len(labels))
        texts = texts[:min_len]
        labels = labels[:min_len]
    
    return texts, labels


def run_intent_mining(texts, initial_distance=0.3, initial_min_samples=16, 
                      delta_distance=0.01, delta_min_samples=1, max_iterations=15, 
                      threshold=300):
    """
    Run ITER-DBSCAN clustering on the texts
    
    Args:
        texts: List of text strings
        initial_distance: Initial distance parameter for DBSCAN
        initial_min_samples: Initial minimum samples parameter for DBSCAN
        delta_distance: Change in distance parameter per iteration
        delta_min_samples: Change in minimum samples parameter per iteration
        max_iterations: Maximum number of iterations
        threshold: Maximum cluster size threshold
        
    Returns:
        Tuple of (cluster_labels, embeddings)
    """
    print(f"Running ITER-DBSCAN on {len(texts)} texts...")
    print(f"Parameters: initial_distance={initial_distance}, initial_min_samples={initial_min_samples}")
    print(f"          delta_distance={delta_distance}, delta_min_samples={delta_min_samples}")
    print(f"          max_iterations={max_iterations}, threshold={threshold}")
    
    # Initialize the model
    model = ITER_DBSCAN(
        initial_distance=initial_distance,
        initial_minimum_samples=initial_min_samples,
        delta_distance=delta_distance,
        delta_minimum_samples=delta_min_samples,
        max_iteration=max_iterations,
        threshold=threshold
    )
    
    # Get embeddings first
    embedding_model = SentenceEmbedding()
    embeddings = embedding_model.getEmbeddings(texts)
    
    # Run clustering
    cluster_labels = model.fit_predict(texts)
    
    return cluster_labels, embeddings


def main():
    parser = argparse.ArgumentParser(description='Intent Mining using ITER-DBSCAN')
    parser.add_argument('--input', '-i', required=True, help='Input data file path')
    parser.add_argument('--text-col', default='message', help='Name of text column (default: text)')
    parser.add_argument('--label-col', default='Intents', help='Name of label column (optional)')
    parser.add_argument('--output', '-o', default='clustering_results_IM.csv', help='Output file path')
    parser.add_argument('--initial-distance', type=float, default=0.3, help='Initial distance parameter')
    parser.add_argument('--initial-min-samples', type=int, default=16, help='Initial minimum samples parameter')
    parser.add_argument('--delta-distance', type=float, default=0.01, help='Distance change per iteration')
    parser.add_argument('--delta-min-samples', type=int, default=1, help='Minimum samples change per iteration')
    parser.add_argument('--max-iterations', type=int, default=15, help='Maximum iterations')
    parser.add_argument('--threshold', type=int, default=300, help='Maximum cluster size threshold')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    texts, true_labels = load_data(args.input, args.text_col, args.label_col)
    print(f"Loaded {len(texts)} texts")
    
    if true_labels:
        print(f"Found {len(set(true_labels))} true label categories")
    
    # Run clustering
    cluster_labels, embeddings = run_intent_mining(
        texts,
        initial_distance=args.initial_distance,
        initial_min_samples=args.initial_min_samples,
        delta_distance=args.delta_distance,
        delta_min_samples=args.delta_min_samples,
        max_iterations=args.max_iterations,
        threshold=args.threshold
    )
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'text': texts,
        'cluster_id': cluster_labels
    })
    
    if true_labels:
        results_df['true_label'] = true_labels
    
    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    
    # Evaluate results if true labels are available
    if true_labels:
        print("\nEvaluating clustering results...")
        evaluator = EvaluateDataset()
        
        # Convert labels to numeric for evaluation
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        numeric_true_labels = le.fit_transform(true_labels)
        
        evaluation_results = evaluator.evaluate_clustering(
            numeric_true_labels, 
            cluster_labels, 
            embeddings
        )
        
        print("\nEvaluation Results:")
        print("=" * 40)
        for metric, value in evaluation_results.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        # Save evaluation results
        evaluator.save_results(evaluation_results, "evaluation_results.txt")
        
        # Analyze clusters
        print("\nCluster Analysis:")
        evaluator.analyze_clusters(texts, cluster_labels, top_n=3)
        
        # Plot cluster distribution
        evaluator.plot_cluster_distribution(cluster_labels, "ITER-DBSCAN Clustering Results")
    
    else:
        print("\nNo true labels provided. Skipping evaluation.")
        print("Cluster Analysis:")
        evaluator = EvaluateDataset()
        evaluator.analyze_clusters(texts, cluster_labels, top_n=3)
        evaluator.plot_cluster_distribution(cluster_labels, "ITER-DBSCAN Clustering Results")


if __name__ == "__main__":
    main()
