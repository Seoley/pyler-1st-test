from typing import List

import torch
from torch import Tensor
from PIL import ImageFile
from transformers import CLIPModel, CLIPProcessor

from .retriever_interface import RetrieverInterface

class ClipModelRetriever(RetrieverInterface):
    """
    CLIP Model class for training multimodal retrieval models based on MM-EMBED.
    """
    def __init__(self, device) -> None:

        self.device = device

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.model_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def processor(self, *args, **kwargs) -> Tensor:
        """
        Process input data (texts and/or images) and convert them into tensors.

        Args:
            texts (List[str], optional): A list of text inputs.
            images (List[ImageFile.ImageFile], optional): A list of image inputs.

        Returns:
            Tensor: The processed tensor converted to the specified device.
                    Returns None if no valid input is provided.
        """
        texts: List[str] = kwargs.get("texts", None)
        images: List[ImageFile.ImageFile] = kwargs.get("images", None)

        if texts and images:
            return self.model_processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(self.device)
        elif texts:
            return self.model_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        elif images:
            return self.model_processor(images=images, return_tensors="pt", padding=True, truncation=True).to(self.device)
        else:
            return None
            
    def text_encoder(self, *args, **kwargs) -> Tensor:
        inputs = self.processor(texts=kwargs.get("texts", None))
        if inputs:
            return self.model.get_text_features(**inputs)
        else:
            return torch.Tensor([]).to(self.device)
    
    def image_encoder(self, *args, **kwargs) -> Tensor:
        inputs = self.processor(images=kwargs.get("images", None))
        if inputs:
            return self.model.get_image_features(**inputs)
        else:
            return torch.Tensor([]).to(self.device)
        
    def multimodal_encoder(self, *args, **kwargs) -> Tensor:
        inputs = self.processor(texts=kwargs.get("texts", None), images=kwargs.get("images", None))
        if inputs:
            return self.model(**inputs)
        else:
            return torch.Tensor([]).to(self.device)