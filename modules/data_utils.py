import random
import os
import pandas as pd
import zipfile
from typing import Callable, Tuple

from modules.utils import download_file
from modules.mining_multimodal import MultimodalHardMiner
from modules.logger import get_logger

logger = get_logger("my_module")

DATASET_DIR = "flickr8k"

DATASET_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
CAPTIONS_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"


def load_flickr8k_dataset() -> Tuple[str, str]:
    """
    Lodas flickr8k dataset and captions.
    If flickr8k dataset does not exist, it downloads and extracts the dataset.

    Returns:
        Tuple[str, str]: Image directory and caption file path.
    """
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR, exist_ok=True)

        dataset_zip = os.path.join(DATASET_DIR, "Flickr8k_Dataset.zip")
        captions_zip = os.path.join(DATASET_DIR, "Flickr8k_text.zip")

        download_file(DATASET_URL, dataset_zip)
        download_file(CAPTIONS_URL, captions_zip)

        with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
            zip_ref.extractall(DATASET_DIR)
        with zipfile.ZipFile(captions_zip, "r") as zip_ref:
            zip_ref.extractall(DATASET_DIR)

    image_dir = os.path.join(DATASET_DIR, "Flicker8k_Dataset")
    captiopn_file = os.path.join(DATASET_DIR, "Flickr8k.token.txt")
    return image_dir, captiopn_file

def generate_dummy_query_set(data_table: pd.DataFrame) -> pd.DataFrame:
    """
    It is a dataset consist of query and positive candidate based on flickr8k dataset.
    Since this is dummy data for testing, the positive candidate was randomly selected 
    from texts and images with the same label.

    Args:
        data_table (pd.DataFrame): It is reference table of dataset and contains three columns: text, image, label

    Returns:
        pd.DataFrame: Query set for multimodal retrieval model.
    """
    query_textset = []
    query_labels = []
    query_document_set = []
    query_document_type_set = []

    label = None
    for idx, row in data_table.iterrows():
        if label != row["label"]:
            query_textset.append(idx)
            label = row["label"]
            query_labels.append(label)

            if random.randint(0, 1) == 0:
                query_document_set.append(idx+1)
                query_document_type_set.append("text")
            else:
                query_document_set.append(idx+1)
                query_document_type_set.append("image")

    query_set = pd.DataFrame(
                        {
                            "text": query_textset,
                            "document": query_document_set,
                            "document_type": query_document_type_set,
                            "label": query_labels
                        }
                    )

    return query_set

def generate_dummy_text_query_set(data_table: pd.DataFrame) -> pd.DataFrame:
    """
    It is a dataset consist of query and positive candidates based on flickr8k dataset for
    text-to-text fine-tuning. Since this is dummy data for testing, the positive candidate 
    was randomly selected from texts with the same label. The positive candidates of this 
    function are only text data.

    Args:
        data_table (pd.DataFrame): It is reference table of dataset and contains three columns: text, image, label

    Returns:
        pd.DataFrame: Query set for text-to-text fine-tuning.
    """
    query_textset = []
    query_labels = []
    query_document_set = []
    query_document_type_set = []

    label = None
    for idx, row in data_table.iterrows():
        if label != row["label"]:
            query_textset.append(idx)
            label = row["label"]
            query_labels.append(label)

            query_document_set.append(idx+1)
            query_document_type_set.append("text")

    query_set = pd.DataFrame(
                        {
                            "text": query_textset,
                            "document": query_document_set,
                            "document_type": query_document_type_set,
                            "label": query_labels
                        }
                    )

    return query_set

def generate_train_index_table(
        data_table: pd.DataFrame,
        query_set: pd.DataFrame,
        text_encoder: Callable=None,
        image_encoder: Callable=None
    ):
    """
    It makes train index table for training multimodal retrieval model.
    For data and memory management, all training data is structured as indexes referencing the data_table.

    Args:
        data_table (pd.DataFrame): It is reference table of dataset and contains three columns: text, image, label
        query_set (pd.DataFrame): It is query and positive candidate table. It contains three columns: text, document, document_type
                                    The columns of text and document is index of data_table.
                                    The document column is positive candidate index and document_type is type of the document(text or image) 
        text_encoder (Callable): Text encoder model.
        image_encoder (Callable): Image encoder model.
    
    Returns:
        pd.DataFrame: Train index table for training multimodal retrieval model.

    Examples: 
        >>> data_table = pd.DataFrame(
                            "text": ["What is the capital of South Korea?", "What is the color of apple?", ...],
                            "image": ["image_path1", "image_path2", ...],
                            "label": [0, 1, 2, ...]
                        )
        >>> query_set = pd.DataFrame(
                            "text": [0, 1, 4, 6, ...],
                            "document": [1, 2, 5, 7, ...],
                            "document_type": ["text", "image", "text", "image", ...]
                        )
        >>> train_index_table = generate_train_index_table(data_table, query_set, text_encoder, image_encoder)
        >>> train_index_table.iloc[0]
        text                0
        document            1
        document_type       "text"
        negative_indices    { "text": [2, 3, 4, 5, 6], "image": [7, 8, 9, 10, 11] }
    
    Notes:
        It used for modality-aware hard negative mining and text-to-text fine-tuning.
        When fine-tuning, image_encoder is not necessary. So the default value of image_encoder is None.
    """
    multimodal_hard_miner = MultimodalHardMiner()
    multimodal_hard_miner.load(text_encoder, image_encoder)
    negative_indices_set = []
    
    for _, row in query_set.iterrows():
        query_index = row["text"]
        document_type = row["document_type"]
        negative_indices = multimodal_hard_miner.get_negative_samples(data_table, idx=query_index, document_type=document_type)
        negative_indices_set.append(negative_indices)

    train_index_table = pd.DataFrame(
                                {
                                    "query": query_set["text"],
                                    "document": query_set["document"],
                                    "document_type": query_set["document_type"],
                                    "negative_indices": negative_indices_set
                                }
        )
    
    return train_index_table