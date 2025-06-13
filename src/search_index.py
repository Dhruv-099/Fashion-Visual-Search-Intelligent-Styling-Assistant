# src/search_index.py
# Handles building, saving, loading, and searching the Faiss index.

import faiss
import numpy as np
import pickle
import os

class SearchIndex:
    def __init__(self, dimension=None):
        """
        Initializes the SearchIndex.

        Args:
            dimension (int, optional): The dimension of the feature vectors.
                                       Required if building a new index.
        """
        self.index = None
        self.index_to_product_id = {}
        self.dimension = dimension

    def build_index(self, vectors: np.ndarray, product_ids: np.ndarray):
        """
        Builds a Faiss index from feature vectors.

        Args:
            vectors (np.ndarray): A NumPy array of feature vectors (float32).
            product_ids (np.ndarray): A NumPy array of product IDs corresponding to the vectors.
        """
        if self.dimension is None or self.dimension != vectors.shape[1]:
             self.dimension = vectors.shape[1]
             print(f"Setting index dimension to {self.dimension}")

        # For a prototype, IndexFlatL2 is simple. For larger datasets, consider HNSW or IVF.
        # vectors should be float32
        if vectors.dtype != np.float32:
             print(f"Warning: Converting vectors from {vectors.dtype} to float32 for Faiss.")
             vectors = vectors.astype('float32')

        self.index = faiss.IndexFlatL2(self.dimension)
        print(f"Adding {vectors.shape[0]} vectors to Faiss index...")
        self.index.add(vectors) # Add the vectors to the index
        print(f"Faiss index total items: {self.index.ntotal}")

        # Create the mapping from index ID to product_id
        self.index_to_product_id = {i: product_id for i, product_id in enumerate(product_ids)}

    def search(self, query_vector: np.ndarray, k=10):
        """
        Searches the index for the k nearest neighbors of a query vector.

        Args:
            query_vector (np.ndarray): The feature vector of the query image (float32).
            k (int): The number of nearest neighbors to retrieve.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: Array of shape (1, k) with the indices of the nearest neighbors.
                - np.ndarray: Array of shape (1, k) with the distances to the nearest neighbors.
        """
        if self.index is None:
            raise RuntimeError("Faiss index is not built or loaded.")

        # Query vector must be float32 and reshaped to [1, dimension]
        query_vector = query_vector.astype('float32').reshape(1, -1)

        # D = Distances, I = Indices
        D, I = self.index.search(query_vector, k)
        return I, D

    def save_index(self, index_path: str, mapping_path: str):
        """Saves the Faiss index and the index-to-product_id mapping."""
        if self.index is None:
             print("Warning: No index to save.")
             return
        faiss.write_index(self.index, index_path)
        with open(mapping_path, "wb") as f:
            pickle.dump(self.index_to_product_id, f)
        print(f"Index saved to {index_path} and mapping to {mapping_path}")

    def load_index(self, index_path: str, mapping_path: str):
        """Loads the Faiss index and the index-to-product_id mapping."""
        if not os.path.exists(index_path) or not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Index or mapping file not found. Expected: {index_path}, {mapping_path}")

        self.index = faiss.read_index(index_path)
        with open(mapping_path, "rb") as f:
            self.index_to_product_id = pickle.load(f)
        self.dimension = self.index.d # Set dimension from loaded index
        print(f"Index loaded from {index_path} with dimension {self.dimension}")

# Example Usage (in main.py and app.py):
# Build:
# search_index = SearchIndex()
# search_index.build_index(vectors, product_ids)
# search_index.save_index(...)

# Load:
# search_index = SearchIndex()
# search_index.load_index(...)

# Search:
# indices, distances = search_index.search(query_vector, k=10)