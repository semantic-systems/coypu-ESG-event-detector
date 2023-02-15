import abc
import copy
from pathlib import Path

import torch
from typing import Union, Dict, List
from omegaconf import DictConfig
from torch.nn import Module, ModuleList
from transformers import PreTrainedModel
from models.event_detection.src.data_augmenters.tweet_normalizer import clean_up_tokenization, normalizeTweet


class SequenceClassification(Module):
    def __init__(self, cfg: DictConfig):
        super(SequenceClassification, self).__init__()
        self.cfg = cfg
        self.encoder = self.instantiate_encoder()
        self.feature_transformer = self.instantiate_feature_transformer()
        self.classification_head = self.instantiate_classification_head()

    @abc.abstractmethod
    def forward(self, **kwargs):
        """
        steps:
        x = self.encoder(x)
        x = self.feature_transformer(x)
        x = self.classification_head(x)

        """
        raise NotImplementedError

    @abc.abstractmethod
    def instantiate_encoder(self):
        raise NotImplementedError

    @abc.abstractmethod
    def instantiate_feature_transformer(self):
        raise NotImplementedError

    @abc.abstractmethod
    def instantiate_classification_head(self):
        raise NotImplementedError

    @staticmethod
    def trim_encoder_layers(encoder: PreTrainedModel, num_layers_to_keep: Union[int, str]) -> PreTrainedModel:
        if num_layers_to_keep == "full":
            return encoder
        else:
            full_layers = encoder.encoder.layer
            trimmed_layers = ModuleList()

            # Now iterate over all layers, only keeping only the relevant layers.
            for i in range(num_layers_to_keep):
                trimmed_layers.append(full_layers[i])

            # create a copy of the model, modify it with the new list, and return
            trimmed_encoder = copy.deepcopy(encoder)
            trimmed_encoder.encoder.layer = trimmed_layers
            return trimmed_encoder

    @staticmethod
    def freeze_encoder(encoder: PreTrainedModel, layers_to_freeze: Union[str, int]) -> PreTrainedModel:
        if layers_to_freeze == "none":
            return encoder
        elif layers_to_freeze == "all":
            for param in encoder.parameters():
                param.requires_grad = False
            return encoder
        elif isinstance(layers_to_freeze, int) and layers_to_freeze <= len(encoder.encoder.layer) - 1:
            modules = [encoder.embeddings, *encoder.encoder.layer[:layers_to_freeze]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
            return encoder
        else:
            raise ValueError(f"Currently, only 'all', 'none' and integer (<=num_transformer_layers) "
                             f"are valid value for freeze_transformer_layer")

    def normalize(self, text: List[str]) -> List[str]:
        normalized_text = text
        if self.cfg.data.name in ["tweet_eval"]:
            normalized_text: List[str] = [normalizeTweet(tweet) for tweet in text]
            normalized_text = [clean_up_tokenization(tweet) for tweet in normalized_text]
        return normalized_text

    def preprocess(self, batch):
        return self.tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")
        # remove max_length arg to use the model's original value for that

    def save_model(self, path: Path, index_label_map: Dict):
        torch.save({
            'config': self.cfg,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'index_label_map': index_label_map
        }, path)

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")