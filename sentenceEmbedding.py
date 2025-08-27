from sentence_transformers import SentenceTransformer
import numpy as np


class SentenceEmbedding(object):
    """Modern sentence embedding class using sentence-transformers"""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = None
        self.model_name = model_name
        self.loadModel()

    def loadModel(self):
        if self.model is None:
            print(f'Loading Sentence Transformer model: {self.model_name}....')
            self.model = SentenceTransformer(self.model_name)
            print('Model Loaded.')

    def embed(self, input_):
        if isinstance(input_, str):
            input_ = [input_]
        embeddings = self.model.encode(input_, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    def getEmbeddings(self, data):
        """
        Get embeddings for a list of texts
        Args:
            data: List of text strings
        Returns:
            List of embedding vectors
        """
        if not data:
            return []
            
        # Process in batches for memory efficiency
        batch_size = 1000
        all_embeddings = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_embeddings = self.embed(batch)
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings


# Global instance for backward compatibility
obj = SentenceEmbedding()

def getEmbeddings(data):
    """
    Global function for backward compatibility
    """
    return obj.getEmbeddings(data)
