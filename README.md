# Intent Mining with ITER-DBSCAN

This is a replication of the "Intent Mining from past conversations for Conversational Agent". The implementation has been updated to use modern, supported packages instead of deprecated ones.

## Overview

This repository implements ITER-DBSCAN (Iterative DBSCAN) for unbalanced data clustering, specifically designed for conversational intent mining from utterances. The algorithm iteratively adapts DBSCAN parameters to find high to low density clusters.

**Original Paper**: [Intent Mining from past conversations for Conversational Agent](https://www.aclweb.org/anthology/2020.coling-main.366/)

**Original Repository**: https://github.com/ajaychatterjee/IntentMining

## Key Features

- **Modern Dependencies**: Uses `sentence-transformers` instead of deprecated `tensorflow_hub`
- **ITER-DBSCAN Algorithm**: Iteratively adapts DBSCAN parameters for unbalanced data clustering
- **Sentence Embeddings**: High-quality sentence representations using state-of-the-art transformer models
- **Comprehensive Evaluation**: Multiple clustering evaluation metrics
- **Easy-to-use API**: Simple interface for running intent mining on your data

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd intentMining
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- `sentence-transformers>=2.2.0` - Modern sentence embedding models
- `scikit-learn>=1.0.0` - Machine learning utilities
- `hdbscan>=0.8.29` - Hierarchical DBSCAN clustering
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `torch>=1.13.0` - PyTorch backend for transformers
- `transformers>=4.20.0` - Hugging Face transformers library

## Quick Start

### Basic Usage

```python
from ITER_DBSCAN import ITER_DBSCAN
from sentenceEmbedding import SentenceEmbedding

# Initialize the model
model = ITER_DBSCAN(
    initial_distance=0.3,
    initial_minimum_samples=16,
    delta_distance=0.01,
    delta_minimum_samples=1,
    max_iteration=15,
    threshold=300
)

# Your text data
texts = ["How do I reset my password?", "I forgot my password", ...]

# Run clustering
cluster_labels = model.fit_predict(texts)
```

### Command Line Usage

```bash
# Basic clustering
python main.py --input your_data.csv --text-col text_column

# With evaluation (if you have true labels)
python main.py --input your_data.csv --text-col text_column --label-col label_column

# Custom parameters
python main.py --input your_data.csv --text-col text_column \
    --initial-distance 0.2 --initial-min-samples 10 --max-iterations 20
```

### Example Script

Run the included example to see the system in action:

```bash
python example.py
```

## API Reference

### ITER_DBSCAN Class

**Parameters:**
- `initial_distance` (float): Initial distance for cluster creation (default: 0.10)
- `initial_minimum_samples` (int): Initial minimum sample count (default: 20)
- `delta_distance` (float): Change in distance per iteration (default: 0.01)
- `delta_minimum_samples` (int): Change in minimum samples per iteration (default: 1)
- `max_iteration` (int): Maximum iterations (default: 5)
- `threshold` (int): Maximum cluster size threshold (default: 300)
- `features` (str): Set to "precomputed" if using pre-computed features

**Methods:**
- `fit_predict(X)`: Compute cluster labels for input data
- `compute(data)`: Internal clustering computation

### SentenceEmbedding Class

**Parameters:**
- `model_name` (str): Name of the sentence transformer model (default: "all-MiniLM-L6-v2")

**Methods:**
- `getEmbeddings(texts)`: Generate embeddings for a list of texts
- `embed(text)`: Generate embedding for a single text

### EvaluateDataset Class

**Methods:**
- `evaluate_clustering(true_labels, predicted_labels, embeddings)`: Evaluate clustering results
- `plot_cluster_distribution(labels, title)`: Visualize cluster distribution
- `analyze_clusters(texts, labels, top_n)`: Analyze cluster contents
- `save_results(results, filename)`: Save evaluation results to file

## Data Format

The system accepts data in the following formats:

### CSV/Excel Files
- **Required**: A column containing text data
- **Optional**: A column containing true labels for evaluation

Example CSV structure:
```csv
text,label
"How do I reset my password?",Reset Password
"I forgot my password",Reset Password
"How do I delete my account?",Delete Account
```

### Text Column
- Default column name: `text`
- Can be specified with `--text-col` parameter
- Should contain the conversation utterances or text snippets

### Label Column (Optional)
- Can be specified with `--label-col` parameter
- Used for evaluation and comparison with clustering results

## Algorithm Details

ITER-DBSCAN works by:

1. **Initialization**: Starts with initial DBSCAN parameters
2. **Iterative Clustering**: Runs DBSCAN multiple times with adapted parameters
3. **Parameter Adaptation**: 
   - Distance parameter increases by `delta_distance` each iteration
   - Minimum samples decreases by `delta_minimum_samples` each iteration
4. **Cluster Merging**: Combines clusters from different iterations
5. **Noise Handling**: Marks points that don't fit any cluster as noise (-1)

## Model Selection

The default sentence transformer model is `all-MiniLM-L6-v2`, which provides:
- Good performance on semantic similarity tasks
- Fast inference speed
- Reasonable model size (80MB)

You can change the model by modifying the `SentenceEmbedding` class initialization:

```python
embedding_model = SentenceEmbedding("paraphrase-multilingual-MiniLM-L12-v2")
```

## Performance Tips

1. **Batch Processing**: The system automatically processes embeddings in batches for memory efficiency
2. **Parameter Tuning**: Adjust `initial_distance` and `initial_minimum_samples` based on your data characteristics
3. **Threshold Setting**: Set `threshold` based on expected cluster sizes in your domain
4. **Model Selection**: Choose sentence transformer models based on your language and performance requirements

## Output Files

The system generates several output files:

1. **Clustering Results**: CSV file with text, cluster IDs, and optional true labels
2. **Evaluation Results**: Text file with clustering evaluation metrics
3. **Visualizations**: Cluster distribution plots (if matplotlib is available)

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size in `SentenceEmbedding.getEmbeddings()`
2. **Poor Clustering**: Adjust `initial_distance` and `initial_minimum_samples` parameters
3. **Model Download**: First run will download the sentence transformer model (~80MB)

### Performance Optimization

- Use smaller sentence transformer models for faster processing
- Process data in smaller batches if memory is limited
- Consider using pre-computed embeddings for large datasets

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@inproceedings{chatterjee-sengupta-2020-intent,
    title = "Intent Mining from past conversations for Conversational Agent",
    author = "Chatterjee, Ajay  and
      Sengupta, Shubhashis",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.366",
    pages = "4140--4152",
    abstract = "Conversational systems are of primary interest in the AI community. Organizations are increasingly using chatbot to provide round-the-clock support and to increase customer engagement. Many commercial bot building frameworks follow a standard approach that requires one to build and train an intent model to recognize user input. These frameworks require a collection of user utterances and corresponding intent to train an intent model. Collecting a substantial coverage of training data is a bottleneck in the bot building process. In cases where past conversation data is available, the cost of labeling hundreds of utterances with intent labels is time-consuming and laborious. In this paper, we present an intent discovery framework that can mine a vast amount of conversational logs and to generate labeled data sets for training intent models. We have introduced an extension to the DBSCAN algorithm and presented a density-based clustering algorithm ITER-DBSCAN for unbalanced data clustering. Empirical evaluation on one conversation dataset, six different intent dataset, and one short text clustering dataset show the effectiveness of our hypothesis.",
}
```

## License

This implementation is provided under the same license as the original repository.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the implementation.
