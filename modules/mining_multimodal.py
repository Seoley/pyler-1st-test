"""
This python file contains the implementation of the Multimodal Hard Negative Mining class.
"""
from typing import Dict, List, TypedDict
import random

import torch
from torch import Tensor
import pandas as pd

from .utils import open_image
from .logger import get_logger

logger = get_logger()

class QueryParams(TypedDict):
    idx: int
    query: str

class MultimodalHardMiner:
    """
    Modality-Aware Hard Negative Mining class based on:
    "Modality-Aware Contrastive Learning for Vision-and-Language Navigation."
    
    It generates two types of negative samples:
        1) Incorrect modality (C1): Image samples ranked higher than the labeled positive.
        2) Unsatisfactory information (C2): Text samples ranked lower than the labeled positive.
    """
    def __init__(self, topk_size: int = 80, k: int = 45):
        """
        Args:
            topk_size(int): Number of samples to consider for hard negative mining.
                    The value is 50 in the original paper.
            k(int): Number of thresholds to consider for 
                    incorrect modality (C1) and unsatisfactory information (C2) negatives.
                    The value is 45 in the original paper.

        Attributes:
            text_encoder(Callable): Text encoder model.
            image_encoder(Callable): Image encoder model.
            processor(Callable): Processor for text and image inputs.
        """
        logger.info("Initializing Multimodal Hard Negative Mining Controler.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.topk_size = topk_size
        self.k = k

        self.random_negatvie_size = 20
        
        self.text_encoder = None
        self.image_encoder = None

    def load(self, text_encoder = None, image_encoder = None) -> None:
        """
        If you initialize the class with the constructor, you do not need to call this method.

        Args:
            text_encoder(Callable): Text encoder model.
            image_encoder(Callable): Image encoder model.
        """
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

    def _rank_samples(self, query_embedding: Tensor, sample_embeddings: Tensor) -> List[int]:
        """
        Compute similarity rankings between query and samples.

        Args:
            query_embedding(Tensor): Query embedding(positive candidate)
            sample_embeddings(Tensor): Sample embeddings(Another candidates in document set)
        Returns:
            List[int]: Topk indices of the samples ranked by similarity.
        """
        similarities = torch.cosine_similarity(query_embedding, sample_embeddings, dim=1)
        _, topk_indices = torch.topk(similarities, self.topk_size, largest=True)
        return topk_indices.tolist()
 
    def _generate_text_embeddings(self, samples: List[str]) -> Tensor:
        if self.text_encoder:
            return self.text_encoder(texts = samples)
        else:
            return torch.Tensor([]).to(self.device)

    def _generate_image_embeddings(self, samples: List[str]) -> Tensor:
        if self.image_encoder:
            images = [open_image(sample) for sample in samples]
            return self.image_encoder(images=images)
        else:
            return torch.Tensor([]).to(self.device)

    def _find_index(self, lst: List[int], value: int) -> int:
        """
        Find position of the value in the list.

        Args:
            lst (List[int]): List of indices
            values (int): Value to find in the list

        Return
            int: Position of the value in the list
        """
        return lst.index(value) if value in lst else self.topk_size
  
    def get_negative_samples(self, dataset: pd.DataFrame, idx: int, document_type: str) -> Dict[str, List]:
        hard_negative_samples = self.get_hard_negative_samples(dataset, idx, document_type)
        if hard_negative_samples:
            return hard_negative_samples
        else:
            return self.get_random_negative_samples(dataset, idx)
            
    def get_hard_negative_samples(self, dataset: pd.DataFrame, idx: int, document_type: str) -> Dict[str, List]:
        """
        Get both incorrect modality (C1) and unsatisfactory information (C2) negatives.
        If the document type of the correct answer is an image, C1 consists of text, 
        and C2 consists of images. If the document type is text, C1 consists of images, 
        and C1 consists of text.

        Args:
            dataset (pd.DataFrame): It contains three columns: text, image, label
            idx (int): Query index
            document_type (str): Document type of the query("text" or "image")

        Returns:
            Dict: A dictionary of two lists of indexes of negative samples for text and image.
        
        Example:
            >>> get_hard_negative_samples(dataset, 0, "text")
            {'text': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'image': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
        """
        document_idx = idx

        text_samples = dataset["text"].tolist()
        image_samples = dataset["image"].tolist()

        query_embedding = self._generate_text_embeddings([dataset.loc[document_idx, "text"]])
        text_embeddings = self._generate_text_embeddings(text_samples)
        image_embeddings = self._generate_image_embeddings(image_samples)

        image_start_index = len(text_embeddings)
        embeddings = torch.cat([text_embeddings, image_embeddings], dim=0)
        topk_indices = self._rank_samples(query_embedding, embeddings)
        boundary_index = self._find_index(topk_indices, document_idx)

        text_indices = []
        image_indices = []
        if document_type == "text":
            text_indices = [idx for idx in topk_indices[self.k:] if idx < image_start_index]
            image_indices = [idx - image_start_index for idx in topk_indices[:boundary_index] if idx >= image_start_index]
        else:
            text_indices = [idx for idx in topk_indices[:boundary_index] if idx < image_start_index]
            image_indices = [idx - image_start_index for idx in topk_indices[self.k:] if idx >= image_start_index]
        if text_indices and image_indices:
            return {
                "text": text_indices,
                "image":image_indices
            }
        else:
            return None
    
    def get_random_negative_samples(self, dataset: pd.DataFrame, idx: int) -> Dict[str, List]:
        """
        It generate random negative samples.
        It is used to prevent the result of modality-aware hard negative mining from being None.
        
        Args:
            dataset (pd.DataFrame): It contains three columns: text, image, label
            idx (int): Query index

        Returns:
            Dict: A dictonary of two lists of negative samples for text and image.
        """
        text_indices = []
        image_indices = []
        idx_list = dataset.index.tolist()
    
        idx_list.remove(idx)
        if self.text_encoder:
            text_indices = random.sample(idx_list, min(self.random_negatvie_size, len(idx_list)))
        if self.image_encoder:
            image_indices = random.sample(idx_list, min(self.random_negatvie_size, len(idx_list)))
        return {
                "text": text_indices,
                "image":image_indices
            }
