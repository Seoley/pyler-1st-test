from typing import Protocol

from torch import Tensor

class RetrieverInterface(Protocol):
    def load_model(self, **kwargs) -> None:
        ...
    
    def processor(self, **kwargs) -> Tensor:
        ...

    def text_encoder(self, **kwargs) -> Tensor:
        ...
    
    def image_encoder(self, **kwargs) -> Tensor:
        ...
    
    def multimodal_encoder(self, **kwargs) -> Tensor:
        ...
    