# src/image_downloader.py
# Handles downloading images from URLs.

import pandas as pd
import requests
import os
from tqdm import tqdm # Optional: for progress bar
from tenacity import retry, stop_after_attempt, wait_fixed # For handling transient errors

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2)) # Retry download up to 3 times with 2 sec wait
def download_image(image_url, local_path):
    """Downloads an image from a URL to a local path."""
    if os.path.exists(local_path):
        # print(f"Image already exists: {local_path}") # Can be noisy
        return True # Already downloaded

    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        with open(local_path, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                out_file.write(chunk)
        # print(f"Downloaded: {local_path}") # Can be noisy
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {image_url} to {local_path}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred downloading {image_url}: {e}")
        return False


def download_images_for_dataframe(df, image_dir):
    """
    Downloads images for products listed in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing product data, including 'product_id' and 'feature_image_s3'.
        image_dir (str): Directory where images will be saved.

    Returns:
        pd.DataFrame: The original DataFrame with a 'local_image_path' column added for successfully downloaded images,
                      filtered to only include rows with successful downloads.
    """
    downloaded_paths = {}
    failed_downloads = []

    # Use tqdm to show progress if installed
    iterator = tqdm(df.iterrows(), total=len(df), desc="Downloading Images") if 'tqdm' in globals() else df.iterrows()

    for index, row in iterator:
        product_id = row['product_id']
        image_url = row['feature_image_s3']

        if pd.isna(image_url):
            failed_downloads.append(product_id)
            continue

        local_path = os.path.join(image_dir, f"{product_id}.jpg")

        if download_image(image_url, local_path):
            downloaded_paths[product_id] = local_path
        else:
            failed_downloads.append(product_id)

    print(f"Finished downloading. Failed for {len(failed_downloads)} products.")

    # Filter DataFrame to only include products with successfully downloaded images
    df_downloaded = df[df['product_id'].isin(downloaded_paths.keys())].copy()
    df_downloaded['local_image_path'] = df_downloaded['product_id'].map(downloaded_paths)

    return df_downloaded