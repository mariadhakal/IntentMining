import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class EvaluateDataset:
    """
    Evaluation class for clustering results
    """
    
    def __init__(self):
        pass
    
    def evaluate_clustering(self, true_labels, predicted_labels, embeddings=None):
        """
        Evaluate clustering results using multiple metrics
        
        Args:
            true_labels: Ground truth cluster labels
            predicted_labels: Predicted cluster labels from clustering algorithm
            embeddings: Feature embeddings for silhouette score calculation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        results = {}
        
        # Remove noise points (-1) for evaluation
        valid_mask = predicted_labels != -1
        if valid_mask.sum() > 0:
            valid_true = true_labels[valid_mask]
            valid_pred = predicted_labels[valid_mask]
            
            # Adjusted Rand Index
            results['adjusted_rand_score'] = adjusted_rand_score(valid_true, valid_pred)
            
            # Normalized Mutual Information
            results['normalized_mutual_info_score'] = normalized_mutual_info_score(valid_true, valid_pred)
            
            # Silhouette Score (if embeddings provided)
            if embeddings is not None:
                valid_embeddings = embeddings[valid_mask]
                if len(np.unique(valid_pred)) > 1:
                    results['silhouette_score'] = silhouette_score(valid_embeddings, valid_pred, metric='cosine')
                else:
                    results['silhouette_score'] = 0.0
            
            # Cluster statistics
            results['n_clusters'] = len(np.unique(valid_pred))
            results['n_noise_points'] = (predicted_labels == -1).sum()
            results['n_valid_points'] = valid_mask.sum()
            
        else:
            results['adjusted_rand_score'] = 0.0
            results['normalized_mutual_info_score'] = 0.0
            results['silhouette_score'] = 0.0
            results['n_clusters'] = 0
            results['n_noise_points'] = len(predicted_labels)
            results['n_valid_points'] = 0
            
        return results
    
    def plot_cluster_distribution(self, labels, title="Cluster Distribution"):
        """
        Plot the distribution of cluster sizes
        
        Args:
            labels: Cluster labels
            title: Plot title
        """
        # Remove noise points
        valid_labels = labels[labels != -1]
        
        if len(valid_labels) == 0:
            print("No valid clusters found")
            return
            
        cluster_counts = pd.Series(valid_labels).value_counts().sort_index()
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(cluster_counts)), cluster_counts.values)
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Points')
        plt.title(title)
        plt.xticks(range(len(cluster_counts)), cluster_counts.index)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"Cluster distribution:")
        print(f"Number of clusters: {len(cluster_counts)}")
        print(f"Total points: {len(valid_labels)}")
        print(f"Noise points: {(labels == -1).sum()}")
        print(f"Average cluster size: {len(valid_labels) / len(cluster_counts):.2f}")
    
    def analyze_clusters(self, texts, labels, top_n=5):
        """
        Analyze clusters by showing representative examples
        
        Args:
            texts: List of text strings
            labels: Cluster labels
            top_n: Number of examples to show per cluster
        """
        df = pd.DataFrame({'text': texts, 'cluster': labels})
        
        # Remove noise points
        df_valid = df[df['cluster'] != -1].copy()
        
        if len(df_valid) == 0:
            print("No valid clusters found")
            return
            
        # Group by cluster and show examples
        for cluster_id in sorted(df_valid['cluster'].unique()):
            cluster_texts = df_valid[df_valid['cluster'] == cluster_id]['text'].tolist()
            print(f"\nCluster {cluster_id} ({len(cluster_texts)} texts):")
            print("-" * 50)
            
            # Show top_n examples
            for i, text in enumerate(cluster_texts[:top_n]):
                print(f"{i+1}. {text[:100]}{'...' if len(text) > 100 else ''}")
            
            if len(cluster_texts) > top_n:
                print(f"... and {len(cluster_texts) - top_n} more")
    
    def save_results(self, results, filename="clustering_results.txt"):
        """
        Save evaluation results to a file
        
        Args:
            results: Dictionary of evaluation results
            filename: Output filename
        """
        with open(filename, 'w') as f:
            f.write("Clustering Evaluation Results\n")
            f.write("=" * 40 + "\n\n")
            
            for metric, value in results.items():
                if isinstance(value, float):
                    f.write(f"{metric}: {value:.4f}\n")
                else:
                    f.write(f"{metric}: {value}\n")
        
        print(f"Results saved to {filename}")


def evaluate_clustering_simple(true_labels, predicted_labels):
    """
    Simple evaluation function for quick assessment
    
    Args:
        true_labels: Ground truth labels
        predicted_labels: Predicted cluster labels
        
    Returns:
        Dictionary with basic metrics
    """
    evaluator = EvaluateDataset()
    return evaluator.evaluate_clustering(true_labels, predicted_labels)
