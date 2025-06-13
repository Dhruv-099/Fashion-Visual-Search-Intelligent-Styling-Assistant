# src/data_loader.py
# Handles loading and initial processing of raw data.

import pandas as pd

def load_and_sample_data(csv_paths, num_samples=5000):
    """
    Loads data from multiple CSVs, combines them, and samples a subset.

    Args:
        csv_paths (list): List of paths to CSV files.
        num_samples (int): Number of products to sample.

    Returns:
        pd.DataFrame: A DataFrame containing sampled and combined product data.
    """
    all_df = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            all_df.append(df)
            print(f"Loaded {len(df)} items from {path}")
        except FileNotFoundError:
            print(f"Warning: Data file not found at {path}")
            continue
        except Exception as e:
            print(f"Error loading data from {path}: {e}")
            continue

    if not all_df:
        print("No data files loaded.")
        return pd.DataFrame()

    df_combined = pd.concat(all_df, ignore_index=True)
    print(f"Combined data has {len(df_combined)} items.")

    # Basic cleaning/preparation
    df_combined.drop_duplicates(subset=['product_id'], inplace=True)
    print(f"After dropping duplicates: {len(df_combined)} items.")

    # Filter out items with missing image URLs if critical
    df_combined.dropna(subset=['feature_image_s3'], inplace=True)
    print(f"After dropping items with missing images: {len(df_combined)} items.")

    # Sample the data
    if len(df_combined) > num_samples:
        sampled_df = df_combined.sample(n=num_samples, random_state=42).reset_index(drop=True)
        print(f"Sampled down to {len(sampled_df)} items.")
    else:
        sampled_df = df_combined.reset_index(drop=True)
        print(f"Using all {len(sampled_df)} items as it's less than {num_samples}.")

    # Add any other necessary data cleaning/parsing here
    # e.g., parsing price dictionaries if needed later outside the API response formatting

    return sampled_df