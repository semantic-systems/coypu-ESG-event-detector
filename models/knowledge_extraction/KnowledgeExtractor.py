from abc import abstractmethod
import torch
from typing import Union, Text, List


class KnowledgeExtractor(object):
    def __init__(self, path_to_ckpt: str):
        self.tokenizer, self.model = self.instantiate(path_to_ckpt)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @staticmethod
    @abstractmethod
    def instantiate(path_to_ckpt: str):
        pass
        
    def forward(self, text: Union[Text, List[Text]]):
        pass
