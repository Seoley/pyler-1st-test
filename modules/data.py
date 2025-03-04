import os
from typing import List

import pandas as pd
from torch.utils.data import Dataset

from modules.mining_multimodal import MultimodalHardMiner
from modules.models import RetrieverInterface
from modules.logger import get_logger
from .data_utils import load_flickr8k_dataset, generate_train_index_table, generate_dummy_query_set, generate_dummy_text_query_set

logger = get_logger()

class DummyFlickerDataset(Dataset):
    """
    Test dataset class for test using Flicker8k dataset. 
    Using MultimodalHardMiner, it can get positive and negative samples.
    This dataset is designed for image captioning and image-text matching, 
    so it does not contain instructions. Because Flicker8k dataset does not. 
    Additionally, the queries are very simple and short. Therefore, 
    it is mainly used to test universal multimodal retrieval based on 
    modality-aware hard negative mining. Do not used for training.
    """
    def __init__(
        self,
        size: int=16
    ):
        """
        Args:
            size (int): Data_table size. Adjust according to the GPU specifications and memory.

        Attributes:
            raw_dataset (pd.DataFrame): Raw dataset of Flicker8k. It contains two columns: image, captions
            data_table (pd.DataFrame): It contains three columns: text, image, label
                                        It is dummy training dataset refered to raw_dataset.
            text_dataset (List[str]): Text dataset for training
            image_dataset (List[str]): Image dataset for training
            labels (List[int]): Label dataset for training
            image_dir (str): Image directory of Flicker8k dataset
            caption_file (str): Caption file of Flicker8k dataset
        """
        logger.info("Initializing DummyFlickerDataset...")

        self.size: int = size

        self.raw_dataset: pd.DataFrame = None
        self.data_table: pd.DataFrame = None

        self.text_dataset: List[str] = []
        self.image_dataset: List[str] = []
        self.labels: List[int] = []

        self.image_dir, self.caption_file = load_flickr8k_dataset()

        self._load_dataset()
        self._prepare_data_table()

    def _load_dataset(self):
        """
        Load Flicker8k dataset and captions.
        """
        captions_dict = {}

        with open(self.caption_file, "r", encoding="utf-8") as f:
            for line in f:
                img_id, caption = line.strip().split("\t")
                img_id = img_id.split("#")[0]  # # 이후 숫자 제거하여 이미지 파일명 통일

                if img_id not in captions_dict:
                    captions_dict[img_id] = []
                captions_dict[img_id].append(caption)

        self.raw_dataset = pd.DataFrame({"image": list(captions_dict.keys()), "captions": list(captions_dict.values())})

    def _prepare_data_table(self):
        """
        Generate dummy training dataset refered to raw_dataset.
        """
        targets = self.raw_dataset.iloc[:self.size]

        for idx, row in targets.iterrows():
            image_path = os.path.join(self.image_dir, row["image"])
            for caption in row["captions"]:
                self.text_dataset.append(caption)
                self.image_dataset.append(image_path)
                self.labels.append(idx)
        
        self.data_table = pd.DataFrame(
                                    {
                                        "text": self.text_dataset, 
                                        "image": self.image_dataset, 
                                        "label": self.labels
                                    }
                                )
        
    def __len__(self):
        return len(self.text_dataset)
    
    def __getitem__(self, idx):
        try:
            text_data = self.data_table.iloc[idx]["text"]
            image_data = self.data_table.iloc[idx]["image"]
            label = self.data_table.iloc[idx]["label"]
            
            return text_data, image_data, label

        except (IndexError, FileNotFoundError) as e:
            print(f"Error loading data at index {idx}: {e}")
            return None


class DummyHardNegativeDataset(DummyFlickerDataset):
    """
    Dummy dataset class based on Flicker8k dataset for testing modality-aware hard negative mining.
    For data and memory management, all training data is structured as indexes referencing 
    the source dataset(DummyFlickerDataset).
    """
    def __init__(
        self,
        model: RetrieverInterface,
        size: int=16
    ):
        """
        Args:
            model (RetrieverInterface): Multimodal retrieval model. It have to contain text_encoder and image_encoder.
            size (int): Dataset size. 

        Attributes:
            text_encoder (Callable): Text encoder of the model
            image_encoder (Callable): Image encoder of the model
            multimodal_hard_miner (MultimodalHardMiner): Hard negative miner for multimodal retrieval
            train_index_table (pd.DataFrame): It has four columns(query, document, document_type, negative_indices)

        Examples: 
            >>> train_index_table = generate_train_index_table(data_table, query_set, text_encoder, image_encoder)
            >>> train_index_table.iloc[0]
            text                0
            document            1
            document_type       "text"
            negative_indices    { "text": [2, 3, 4, 5, 6], "image": [7, 8, 9, 10, 11] }

        Note:
            negative_indices column is used for hard negative mining.
            For example, if document_type is text, negative_indices["text"] is used for unsatisfactory information(C1),
            and negative_indices["image"] is used for incorrect modality(C2).
            When training, you can get negative indices and use them to get negative samples from dataset.
        """
        super().__init__(size)
        logger.info("Initializing DummyHardNegativeDataset...")

        self.text_encoder = model.text_encoder
        self.image_encoder = model.image_encoder
        
        self.multimodal_hard_miner = MultimodalHardMiner()
        self.multimodal_hard_miner.load(self.text_encoder, self.image_encoder)
        
        self.train_index_table: pd.DataFrame = None

        self._generate_query_set()
        self._generate_train_index_table()

    def _generate_query_set(self):
        """
        It is a dataset consist of query and positive candidate based on flickr8k dataset 
        using function .data_utils.generate_dummy_query_set()
        Since this is dummy data for testing, the positive candidate was randomly 
        selected from texts and images with the same label.
        """
        self.query_set = generate_dummy_query_set(self.data_table)

    def _generate_train_index_table(self):
        """
        It makes train index table for training multimodal retrieval model. 
        For data and memory management, all training data is structured as indexes referencing the data_table.
        """
        self.train_index_table = generate_train_index_table(self.data_table, self.query_set, self.text_encoder, self.image_encoder)
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        try:
            query = self.train_index_table.iloc[idx]["query"]
            document = self.train_index_table.iloc[idx]["document"]
            documnet_type = self.train_index_table.iloc[idx]["document_type"]
            negative_indices = self.train_index_table.iloc[idx]["negative_indices"]

            return query, document, documnet_type, negative_indices

        except (IndexError, FileNotFoundError) as e:
            print(f"Error loading data at index {idx}: {e}")
            return None
        
class DummyTextToTextDataset(DummyFlickerDataset):
    """
    Dummy dataset class based on Flicker8k dataset for testing text-to-text fine-tuning.
    For data and memory management, all training data is structured as indexes referencing 
    the source dataset(DummyFlickerDataset).
    """
    def __init__(
        self,
        model: RetrieverInterface,
        size: int=16
    ):
        """
        Args:
            model (RetrieverInterface): Multimodal retrieval model. It have to contain text_encoder and image_encoder.
            size (int): Dataset size. 

        Attributes:
            text_encoder (Callable): Text encoder of the model
            image_encoder (Callable): Image encoder of the model
            multimodal_hard_miner (MultimodalHardMiner): Hard negative miner for multimodal retrieval
            train_index_table (pd.DataFrame): It has four columns(query, document, document_type, negative_indices)
                                                document_type is only text. So negative_indices["image"] is empty. 
        
        Examples: 
            >>> train_index_table = generate_train_index_table(data_table, query_set, text_encoder, image_encoder)
            >>> train_index_table.iloc[0]
            text                0
            document            1
            document_type       "text"
            negative_indices    { "text": [2, 3, 4, 5, 6], "image": [] }
        """
        super().__init__(size)
        logger.info("Initializing DummyHardNegativeDataset...")

        self._generate_query_set()

        self.text_encoder = model.text_encoder
        
        self.multimodal_hard_miner = MultimodalHardMiner()
        self.multimodal_hard_miner.load(self.text_encoder)
        
        self.train_index_table: pd.DataFrame = None
        self._generate_train_index_table()

    def _generate_query_set(self):
        """
        It is a dataset consist of query and positive candidate based on flickr8k dataset 
        using function .data_utils.generate_dummy_query_set()
        Since this is dummy data for testing, the positive candidate was randomly 
        selected from texts and images with the same label.
        """
        self.query_set = generate_dummy_text_query_set(self.data_table)
        

    def _generate_train_index_table(self):
        """
        It makes train index table for training multimodal retrieval model. 
        For data and memory management, all training data is structured as indexes referencing the data_table.
        Also, document_type is only text. So negative_indices["image"] is empty.
        """
        self.train_index_table = generate_train_index_table(self.data_table, self.query_set, self.text_encoder)
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        try:
            query = self.train_index_table.iloc[idx]["query"]
            document = self.train_index_table.iloc[idx]["document"]
            documnet_type = self.train_index_table.iloc[idx]["document_type"]
            negative_indices = self.train_index_table.iloc[idx]["negative_indices"]

            return query, document, documnet_type, negative_indices

        except (IndexError, FileNotFoundError) as e:
            print(f"Error loading data at index {idx}: {e}")
            return None