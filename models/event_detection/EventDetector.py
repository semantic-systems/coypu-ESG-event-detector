import os
from pathlib import Path

import gdown
import torch
from torch import tensor
from typing import Optional, Tuple
from dataclasses import dataclass
from schemes import EventDetectorOutput
from models import BaseComponent
from models.event_detection.src.models.SingleLabelSequenceClassification import SingleLabelSequenceClassification
from stores.ontologies.event_type_wikidata_links_trecis import EVENT_TYPE_WIKIDATA_LINKS
from transformers import logging
logging.set_verbosity_error()


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class InputFeature:
    input_ids: tensor
    attention_mask: tensor
    labels: Optional[tensor] = None


@dataclass
class SingleLabelClassificationForwardOutput:
    loss: Optional[tensor] = None
    prediction_logits: tensor = None
    encoded_features: Optional[tensor] = None
    attentions: Optional[Tuple[tensor]] = None


class EventDetector(BaseComponent):
    def __init__(self, path_to_pretrained_model: str = "/data/event_detector/crisisbert_w_oos_linear.pt"):
        super(EventDetector).__init__()
        self.prepare(path_to_pretrained_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path_to_pretrained_model, map_location=self.device)
        checkpoint['config']['model']["from_pretrained"] = "/data/language_models/CoyPu-CrisisLM-v1"
        self.model = SingleLabelSequenceClassification(checkpoint['config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.index_label_map = checkpoint['index_label_map']
        self.model.to(self.device)

    def forward(self, tweet: str) -> EventDetectorOutput:
        tokenized_text = self.model.tokenizer(tweet, padding=True, truncation=True, return_tensors="pt")
        input_ids: tensor = tokenized_text["input_ids"].to(self.model.device)
        attention_masks: tensor = tokenized_text["attention_mask"].to(self.model.device)
        labels = None
        input_feature: InputFeature = InputFeature(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
        output: SingleLabelClassificationForwardOutput = self.model.forward(input_feature, "test")
        prediction = output.prediction_logits.argmax(1).item()
        event_type = self.index_label_map[str(prediction)]
        wikidata_link = EVENT_TYPE_WIKIDATA_LINKS.get(event_type)
        return EventDetectorOutput(tweet=tweet, event_type=event_type, wikidata_link=wikidata_link)

    @property
    def __version__(self):
        return "2.0.0"

    @staticmethod
    def prepare(path_to_pretrained_model):
        if not Path(path_to_pretrained_model).exists():
            if not Path("/data/event_detector").exists():
                Path("/data/event_detector").mkdir()
            gdown.download(url="https://drive.google.com/file/d/1Hj_s7UfKYOMszQYAYLy0iNFN1qD1wxrH/view?usp=sharing&confirm=t",
                           output=path_to_pretrained_model, fuzzy=True)
        if not Path("/data/language_models/CoyPu-CrisisLM-v1").exists():
            if not Path("/data/language_models/").exists():
                Path("/data/language_models/").mkdir()
            gdown.download_folder(
                url="https://drive.google.com/drive/folders/1u6Mthkr4ffVNSjPn3F_B49axTsCHwRv8?usp=sharing&confirm=t",
                output="/data/language_models/CoyPu-CrisisLM-v1")
