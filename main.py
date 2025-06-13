import pandas as pd
import numpy as np
import os
import pickle
from src.data_loader import load_and_sample_data
from src.image_downloader import download_images_for_dataframe
from src.feature_extractor import FeatureExtractor
from src.search_index import SearchIndex

# --- Configuration ---
DATA_DIR = "data"
IMAGE_DIR = "images"
FEATURES_DIR = "features"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

DRESSES_CSV = os.path.join(DATA_DIR, 'dresses_bd_processed_data.csv')
JEANS_CSV = os.path.join(DATA_DIR, 'jeans_bd_processed_data.csv')
SAMPLED_METADATA_PKL = os.path.join(FEATURES_DIR, 'sampled_product_metadata.pkl')
PRODUCT_FEATURES_PKL = os.path.join(FEATURES_DIR, 'product_features.pkl')
FAISS_INDEX_FILE = os.path.join(FEATURES_DIR, 'product_features.index')
INDEX_ID_MAPPING_PKL = os.path.join(FEATURES_DIR, 'index_to_product_id.pkl')
METADATA_DICT_PKL = os.path.join(FEATURES_DIR, 'product_id_to_metadata.pkl') # Added for quick lookup



if __name__ == "__main__":
    print("--- Starting Data Preprocessing ---")

    # 1. Load and Sample Data
    print("Loading and sampling data...")
    df_combined = load_and_sample_data([DRESSES_CSV, JEANS_CSV])

    # 2. Download Images
    print(f"Attempting to download images for {len(df_combined)} products...")
    df_combined = download_images_for_dataframe(df_combined, image_dir=IMAGE_DIR)
    print(f"Successfully downloaded images for {len(df_combined)} products.")

    # Save sampled metadata with local paths
    df_combined.to_pickle(SAMPLED_METADATA_PKL)
    print(f"Saved sampled metadata with local paths to {SAMPLED_METADATA_PKL}")

    # Prepare metadata dictionary for quick lookup
    metadata_dict = df_combined.set_index('product_id').to_dict(orient='index')
    with open(METADATA_DICT_PKL, "wb") as f:
        pickle.dump(metadata_dict, f)
    print(f"Saved product metadata dictionary to {METADATA_DICT_PKL}")


    # 3. Feature Extraction
    print("Initializing feature extractor...")
    feature_extractor = FeatureExtractor()

    print("Extracting features...")
    product_ids, features = feature_extractor.extract_features_from_dataframe(df_combined)

    feature_df = pd.DataFrame({'product_id': product_ids, 'features': list(features)}) # Store features as list of arrays
    feature_df.to_pickle(PRODUCT_FEATURES_PKL)
    print(f"Saved extracted features to {PRODUCT_FEATURES_PKL}")

    # Create a dict for quick feature vector lookup by product ID
    feature_vectors_dict = dict(zip(product_ids, features))
    # Note: For very large datasets, storing all features in memory like this might not be feasible.
    # In a real system, features would be loaded on demand or kept in a vector database.


    # 4. Build Search Index
    print("Building search index...")
    # Ensure vectors are float32 as required by Faiss
    vectors = np.vstack(feature_df['features'].values).astype('float32')
    product_ids_array = feature_df['product_id'].values

    search_index = SearchIndex(dimension=vectors.shape[1])
    search_index.build_index(vectors, product_ids_array)

    search_index.save_index(FAISS_INDEX_FILE, INDEX_ID_MAPPING_PKL)
    print(f"Saved Faiss index to {FAISS_INDEX_FILE} and mapping to {INDEX_ID_MAPPING_PKL}")

    print("--- Preprocessing Complete ---")