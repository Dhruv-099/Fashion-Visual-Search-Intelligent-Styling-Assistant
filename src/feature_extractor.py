# src/feature_extractor.py
# Handles loading the CNN model and extracting feature vectors.

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import pandas as pd
from tqdm import tqdm # Optional

from .utils import get_image_transform # Import the standard transform from utils

class FeatureExtractor:
    def __init__(self, model_name='resnet50'):
        """
        Initializes the feature extractor with a pre-trained CNN model.

        Args:
            model_name (str): Name of the pre-trained model to use (e.g., 'resnet50').
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load the pre-trained model
        if model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # Use weights parameter in newer torchvision
        elif model_name == 'resnet101':
             self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        # Add other models if needed
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        # Remove the final classification layer
        # The output of the layer before the classifier is the feature vector
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))

        self.model.eval() # Set model to evaluation mode
        self.model.to(self.device) # Move model to the selected device

        self.preprocess = get_image_transform() # Get standard preprocessing

    def extract_features_from_image_path(self, image_path: str):
        """
        Extracts a feature vector from a single image file.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: The feature vector for the image (NumPy array), or None if processing fails.
        """
        img_tensor = self.preprocess(Image.open(image_path).convert('RGB'))
        img_tensor = img_tensor.unsqueeze(0) # Add batch dimension
        img_tensor = img_tensor.to(self.device) # Move tensor to device

        with torch.no_grad():
            features = self.model(img_tensor)

        # Flatten the output features (e.g., from [1, channels, 1, 1] to [channels])
        features = features.squeeze().cpu().numpy()
        return features

    def extract_features_from_dataframe(self, df: pd.DataFrame, batch_size=32):
        """
        Extracts feature vectors for all images listed in a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with a 'local_image_path' column.
            batch_size (int): Number of images to process in each batch.

        Returns:
            tuple: A tuple containing:
                - list: List of product_ids corresponding to the extracted features.
                - list: List of feature vectors (NumPy arrays).
        """
        features_list = []
        product_ids_list = []

        # Use tqdm to show progress if installed
        iterator = tqdm(range(0, len(df), batch_size), desc="Extracting Features") if 'tqdm' in globals() else range(0, len(df), batch_size)

        for i in iterator:
            batch_df = df.iloc[i:i+batch_size]
            batch_images_tensors = []
            batch_product_ids = []

            for index, row in batch_df.iterrows():
                img_path = row.get('local_image_path') # Use .get for safety
                product_id = row['product_id']
                if not img_path or not os.path.exists(img_path):
                    # Skip if no valid local path
                    continue

                try:
                    img_tensor = self.preprocess(Image.open(img_path).convert('RGB'))
                    batch_images_tensors.append(img_tensor)
                    batch_product_ids.append(product_id)
                except Exception as e:
                    print(f"Error processing image {img_path} for feature extraction: {e}")


            if not batch_images_tensors:
                continue

            # Stack tensors to create a batch and move to device
            batch_tensor = torch.stack(batch_images_tensors).to(self.device)

            # Get features
            with torch.no_grad():
                batch_features = self.model(batch_tensor)

            # Flatten features and move back to CPU
            batch_features = batch_features.squeeze().cpu().numpy()

            features_list.extend(batch_features)
            product_ids_list.extend(batch_product_ids)

        return product_ids_list, features_list

# Example Usage (in main.py):
# feature_extractor = FeatureExtractor()
# product_ids, features = feature_extractor.extract_features_from_dataframe(sampled_df)