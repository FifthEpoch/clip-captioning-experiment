import clip
import torch

from numpy import ndarray
from typing import List
from PIL import Image

import os
from chromadb import Client, Settings
from typing import List

from caption_predictor import get_caption_prediction_from_embeddings, get_caption_prediction_from_image


class DirectionalClipTransformEmbeddingFunction:

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu"):
        self.device = device  # Store the specified device for model execution
        self.model, self.preprocess = clip.load(model_name, self.device)

    def __call__(self, docs: dict[str, str]):
        # Define a method that takes a list of image file paths (docs) and their corresponding captions (caps) as input

        # Create an empty list to store the text embeddings from the corresponding captions (caps)
        list_of_embeddings = []
        img_paths = list(docs.keys())
        for p in img_paths:
            # Open and load an image from the provided path, resize
            image = Image.open(p)
            image = image.resize((224, 224))

            # get ideal caption for image
            ideal_cap = docs[p]

            img_embeddings = self.get_img_embeddings(image)
            target_text_embeddings = self.get_text_embeddings(ideal_cap)

            delta_embeddings = target_text_embeddings - img_embeddings

            list_of_embeddings.append(delta_embeddings)

        return list_of_embeddings

    def get_img_embeddings(self, img: Image.Image):
        # Preprocess the image and move it to the specified device
        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Compute the image embeddings using the CLIP model and convert them to NumPy arrays
            img_embeddings = self.model.encode_image(image_input).cpu().detach().numpy()
        return img_embeddings[0]

    def get_text_embeddings(self, text: str):
        # Define a method that takes a text string as input
        text_token = clip.tokenize(text)  # Tokenize the input text
        with torch.no_grad():
            # Compute the text embeddings using the CLIP model and convert them to NumPy arrays
            text_embeddings = self.model.encode_text(text_token).cpu().detach().numpy()
        return text_embeddings[0]


