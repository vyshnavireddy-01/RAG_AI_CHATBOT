from sentence_transformers import SentenceTransformer
import numpy as np
import config

# Load embedding model once
model = SentenceTransformer(config.MODEL_NAME)


def create_embeddings(texts):
    """
    Convert text chunks into vector embeddings
    """

    if not texts:
        return np.array([])

    embeddings = model.encode(texts)

    return np.array(embeddings)