# app.py
# FastAPI application for the visual search and outfit recommendation API.

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import pickle
import numpy as np

# Import modules from src
from src.feature_extractor import FeatureExtractor
from src.search_index import SearchIndex
from src.outfit_recommender import OutfitRecommender

# --- Configuration ---
FEATURES_DIR = "features"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

PRODUCT_FEATURES_PKL = os.path.join(FEATURES_DIR, 'product_features.pkl')
FAISS_INDEX_FILE = os.path.join(FEATURES_DIR, 'product_features.index')
INDEX_ID_MAPPING_PKL = os.path.join(FEATURES_DIR, 'index_to_product_id.pkl')
METADATA_DICT_PKL = os.path.join(FEATURES_DIR, 'product_id_to_metadata.pkl')


# --- Load Precomputed Assets ---
# These assets are computed by running main.py
print("Loading precomputed assets...")
try:
    feature_extractor = FeatureExtractor() # Initialize feature extractor model

    search_index = SearchIndex()
    search_index.load_index(FAISS_INDEX_FILE, INDEX_ID_MAPPING_PKL)
    print(f"Loaded Faiss index with {search_index.index.ntotal} items.")

    with open(METADATA_DICT_PKL, "rb") as f:
        metadata_dict = pickle.load(f)
    print(f"Loaded metadata for {len(metadata_dict)} products.")

    # Load feature vectors dictionary (needed for outfit recommendations)
    # This can consume significant memory for large datasets
    feature_df = pd.read_pickle(PRODUCT_FEATURES_PKL)
    feature_vectors_dict = dict(zip(feature_df['product_id'], feature_df['features']))
    print(f"Loaded {len(feature_vectors_dict)} feature vectors into memory.")

    outfit_recommender = OutfitRecommender(metadata_dict, feature_vectors_dict)

except FileNotFoundError as e:
    print(f"Error loading assets: {e}")
    print("Please ensure you have run main.py first to generate the assets.")
    # Exit or raise error if assets are critical and missing
    metadata_dict = {}
    feature_vectors_dict = {}
    search_index = None # Indicate index is not loaded
    feature_extractor = None # Indicate extractor is not loaded
    outfit_recommender = None # Indicate recommender is not loaded
except Exception as e:
    print(f"An unexpected error occurred loading assets: {e}")
    metadata_dict = {}
    feature_vectors_dict = {}
    search_index = None
    feature_extractor = None
    outfit_recommender = None


# --- FastAPI App ---
app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Fashion Visual Search & Intelligent Styling Assistant API"}

@app.post("/search")
async def search_fashion(file: UploadFile = File(...)):
    """
    Visual search endpoint: Upload an image and find similar fashion items.
    """
    if search_index is None or feature_extractor is None:
         raise HTTPException(status_code=503, detail="System assets not loaded. Please run preprocessing.")

    # Save the uploaded file temporarily
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract features from the query image
        query_vector = feature_extractor.extract_features_from_image_path(temp_file_path)

        # Search the index
        similar_items_indices, distances = search_index.search(query_vector, k=10)

        results = []
        # Map indices back to product IDs and retrieve metadata
        for i, dist in zip(similar_items_indices[0], distances[0]):
            if i == -1: # Handle cases where index might not return enough results
                continue
            product_id = search_index.index_to_product_id.get(i)
            if product_id and product_id in metadata_dict:
                metadata = metadata_dict[product_id]
                 # Simple conversion to a score 0-1 (lower distance = higher score)
                similarity_score = 1 / (1 + dist)
                results.append({
                    'product_id': product_id,
                    'similarity_score': float(similarity_score), # Ensure JSON serializable
                    'product_name': metadata.get('product_name'),
                    'brand': metadata.get('brand'),
                    # Format selling_price safely
                    'selling_price': metadata.get('selling_price', {}).get('value') if isinstance(metadata.get('selling_price'), dict) else None,
                    'currency': metadata.get('selling_price', {}).get('currency') if isinstance(metadata.get('selling_price'), dict) else None,
                    'feature_image_s3': metadata.get('feature_image_s3'),
                    'pdp_url': metadata.get('pdp_url')
                })

        # Sort by similarity score descending
        results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)

        return JSONResponse(content=results)

    except Exception as e:
        print(f"API Search Error: {e}")
        # Return a specific HTTP status code and detail for client-side handling
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
             os.remove(temp_file_path)


@app.get("/outfit/{product_id}")
async def get_outfit_recommendations(product_id: str):
    """
    Outfit recommendation endpoint: Get outfit suggestions for a given product ID.
    """
    if outfit_recommender is None:
        raise HTTPException(status_code=503, detail="Outfit recommender not initialized. Please run preprocessing.")

    if product_id not in metadata_dict:
         raise HTTPException(status_code=404, detail=f"Product ID '{product_id}' not found in inventory.")

    try:
        # Perform outfit recommendation
        outfit_recs = outfit_recommender.recommend(product_id, k_per_category=5) # Get top 5 per category

        # Format results for JSON response
        formatted_recs = {}
        for category, items in outfit_recs.items():
             formatted_items = []
             for item in items:
                 formatted_items.append({
                     'product_id': item['product_id'],
                     'compatibility_score': float(item['compatibility_score']),
                     'product_name': item['metadata'].get('product_name'),
                     'brand': item['metadata'].get('brand'),
                     # Format selling_price safely
                     'selling_price': item['metadata'].get('selling_price', {}).get('value') if isinstance(item['metadata'].get('selling_price'), dict) else None,
                     'currency': item['metadata'].get('selling_price', {}).get('currency') if isinstance(item['metadata'].get('selling_price'), dict) else None,
                     'feature_image_s3': item['metadata'].get('feature_image_s3'),
                     'pdp_url': item['metadata'].get('pdp_url')
                 })
             formatted_recs[category] = formatted_items

        return JSONResponse(content=formatted_recs)

    except Exception as e:
        print(f"API Outfit Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# To run this app:
# 1. Make sure you have run `python main.py` first.
# 2. Activate your uv environment: `source .venv/bin/activate`
# 3. Run the server: `uvicorn app:app --reload`