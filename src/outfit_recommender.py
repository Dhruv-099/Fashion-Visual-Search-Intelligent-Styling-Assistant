# src/outfit_recommender.py
# Handles generating outfit recommendations.

import numpy as np
import pandas as pd # Potentially useful for data manipulation
from sklearn.metrics.pairwise import cosine_similarity # Or use numpy for dot product if normalized

class OutfitRecommender:
    def __init__(self, metadata_dict: dict, feature_vectors_dict: dict):
        """
        Initializes the OutfitRecommender.

        Args:
            metadata_dict (dict): Mapping from product_id to product metadata.
            feature_vectors_dict (dict): Mapping from product_id to feature vector (NumPy array).
        """
        self.metadata_dict = metadata_dict
        self.feature_vectors_dict = feature_vectors_dict
        self.all_product_ids = np.array(list(feature_vectors_dict.keys()))
        self.all_vectors = np.vstack(list(feature_vectors_dict.values())).astype('float32')

        # Precompute category IDs for quick lookup/filtering
        # Assuming 'category_id' exists in metadata_dict
        self.all_category_ids = np.array([
            metadata_dict.get(pid, {}).get('category_id') for pid in self.all_product_ids
        ])

        # Define example complementary categories mapping
        # Replace with actual category IDs and relationships from your data
        self.complementary_categories_map = {
            # Example: searched item category -> [list of categories for outfit suggestions]
            # You need to map actual category_id values here
            1001: [2001, 3001], # e.g., Dresses (1001) -> Shoes (2001), Accessories (3001)
            1002: [5001, 2001], # e.g., Jeans (1002) -> Tops (5001), Shoes (2001)
            # Add more mappings based on your dataset's categories
        }
        print("OutfitRecommender initialized with metadata and features.")
        print("NOTE: Complementary category mapping is example data. Update in OutfitRecommender.py")


    def recommend(self, searched_product_id: str, k_per_category=5):
        """
        Generates outfit recommendations based on a searched product.

        Args:
            searched_product_id (str): The product ID of the item the user searched for.
            k_per_category (int): Number of recommendations per complementary category.

        Returns:
            dict: A dictionary where keys are category names (or IDs) and values are
                  lists of recommended items for that category, ordinally ranked.
        """
        searched_metadata = self.metadata_dict.get(searched_product_id)
        searched_vector = self.feature_vectors_dict.get(searched_product_id)

        if searched_metadata is None or searched_vector is None:
            print(f"Error: Metadata or features not found for product {searched_product_id}")
            return {"error": "Searched product not found or missing data."}

        searched_category_id = searched_metadata.get('category_id')
        if searched_category_id is None:
             print(f"Error: Category ID not found for product {searched_product_id}")
             return {"error": "Searched product missing category information."}


        complementary_category_ids = self.complementary_categories_map.get(searched_category_id, [])

        outfit_suggestions = {}

        # Use cosine similarity for compatibility scoring (assuming features are somewhat comparable)
        # Alternatively, use L2 distance as in search_index, lower distance = higher compatibility
        # Let's use L2 distance for consistency with the search index.
        # Note: This requires iterating through items, which is less efficient than a vector index search.
        # For scalability, filter the main index search results by category or build category-specific indexes.


        for target_category_id in complementary_category_ids:
            print(f"Generating recommendations for target category ID: {target_category_id}")
            # Find indices of products in the target category
            target_indices = np.where(self.all_category_ids == target_category_id)[0]

            if len(target_indices) == 0:
                print(f"No items found for target category ID: {target_category_id}")
                continue

            target_product_ids = self.all_product_ids[target_indices]
            target_vectors = self.all_vectors[target_indices]

            # Calculate L2 distances between the searched item vector and target category vectors
            # Lower distance means higher visual similarity/potential compatibility
            distances = np.linalg.norm(target_vectors - searched_vector, axis=1)

            # Get indices of the k nearest (most similar) items in the target category
            # Use argpartition for efficiency if k is much smaller than target_indices size
            if k_per_category < len(distances):
                 nearest_indices_in_target_vectors = np.argpartition(distances, k_per_category)[:k_per_category]
                 # Ensure we only take the top k if argpartition gives more
                 distances_k = distances[nearest_indices_in_target_vectors]
                 sort_order = np.argsort(distances_k) # Sort the top k
                 nearest_indices_in_target_vectors = nearest_indices_in_target_vectors[sort_order]
            else:
                 # If k is larger than or equal to available items, take all and sort
                 nearest_indices_in_target_vectors = np.argsort(distances)
                 distances_k = distances[nearest_indices_in_target_vectors]


            recommended_items = []
            for idx_in_target_vectors, dist in zip(nearest_indices_in_target_vectors, distances_k):
                rec_product_id = target_product_ids[idx_in_target_vectors]
                rec_metadata = self.metadata_dict.get(rec_product_id)

                if rec_metadata:
                    # Compatibility score: lower distance = higher score (e.g., 1 / (1+dist))
                    compatibility_score = 1 / (1 + dist)
                    recommended_items.append({
                        'product_id': rec_product_id,
                        'compatibility_score': float(compatibility_score), # Ensure JSON serializable
                        'metadata': rec_metadata # Include full metadata for API
                    })

            # Ordinal ranking is achieved by sorting by compatibility_score (descending)
            # This is already done by argsort on distances which are inverse to compatibility
            # recommended_items.sort(key=lambda x: x['compatibility_score'], reverse=True) # Explicit sort if needed

            # Get a display name for the category (simple example)
            category_name = f"Category {target_category_id}" # Replace with lookup if category names are available
            if recommended_items:
                outfit_suggestions[category_name] = recommended_items

        return outfit_suggestions

# Example Usage (in app.py after loading data):
# outfit_recommender = OutfitRecommender(metadata_dict, feature_vectors_dict)
# recs = outfit_recommender.recommend(some_product_id)