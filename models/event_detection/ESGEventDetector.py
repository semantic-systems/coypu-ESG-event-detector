import os
from pathlib import Path

import gdown
import torch
from torch import tensor
from typing import Optional, Tuple
from dataclasses import dataclass
from schemes import EventDetectorOutput
from models import BaseComponent
from stores.ontologies.event_type_wikidata_links_trecis import EVENT_TYPE_WIKIDATA_LINKS
from transformers import logging, AutoTokenizer, BertForSequenceClassification

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


class ESGEventDetector(BaseComponent):
    def __init__(self, path_to_pretrained_model: str = "../../data/event_detector/esg_event_detector"):
        super(ESGEventDetector).__init__()
        self.prepare(path_to_pretrained_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForSequenceClassification.from_pretrained(
                                            path_to_pretrained_model,
                                            num_labels=26, #number of classifications
                                            output_attentions=False, # Whether the model returns attentions weights.
                                            output_hidden_states=False, # Whether the model returns all hidden-states.
                                            )
        self.tokenizer = AutoTokenizer.from_pretrained(path_to_pretrained_model)
        self.model.to(self.device)
        self.model.eval()
        self.label_index_map = {"Business_Ethics": 0,
                                "Data_Security": 1,
                                "Access_And_Affordability": 2,
                                "Business_Model_Resilience": 3,
                                "Competitive_Behavior": 4,
                                "Critical_Incident_Risk_Management": 5,
                                "Customer_Welfare": 6,
                                "Director_Removal": 7,
                                "Employee_Engagement_Inclusion_And_Diversity": 8,
                                "Employee_Health_And_Safety": 9,
                                "Human_Rights_And_Community_Relations": 10,
                                "Labor_Practices": 11,
                                "Management_Of_Legal_And_Regulatory_Framework": 12,
                                "Physical_Impacts_Of_Climate_Change": 13,
                                "Product_Quality_And_Safety": 14,
                                "Product_Design_And_Lifecycle_Management": 15,
                                "Selling_Practices_And_Product_Labeling": 16,
                                "Supply_Chain_Management": 17,
                                "Systemic_Risk_Management": 18,
                                "Waste_And_Hazardous_Materials_Management": 19,
                                "Water_And_Wastewater_Management": 20,
                                "Air_Quality": 21,
                                "Customer_Privacy": 22,
                                "Ecological_Impacts": 23,
                                "Energy_Management": 24,
                                "GHG_Emissions": 25}
        self.index_label_map = {str(value): key for key, value in self.label_index_map.items()}
        self.model.to(self.device)

    def forward(self, tweet: str) -> EventDetectorOutput:
        tokenized_text = self.tokenizer(tweet, padding=True, truncation=True, return_tensors="pt")
        input_ids: tensor = tokenized_text["input_ids"].to(self.model.device)
        attention_masks: tensor = tokenized_text["attention_mask"].to(self.model.device)
        output = self.model.forward(input_ids=input_ids, attention_mask=attention_masks)
        prediction = output.logits.argmax(1).item()
        event_type = self.index_label_map[str(prediction)]
        wikidata_link = EVENT_TYPE_WIKIDATA_LINKS.get(event_type)
        return EventDetectorOutput(tweet=tweet, event_type=event_type, wikidata_link=wikidata_link)

    @property
    def __version__(self):
        return "2.0.0"

    @staticmethod
    def prepare(path_to_pretrained_model):
        URI = "https://drive.google.com/drive/folders/1-0KvLqgmriMXRBo2Hczgi2nNnK_5l6YI?usp=sharing"
        if not Path(path_to_pretrained_model).exists():
            if not Path("/data/event_detector").exists():
                Path("/data/event_detector").mkdir()
            gdown.download_folder(url=URI,
                                  output=path_to_pretrained_model)


if __name__ == '__main__':
    event_detector = ESGEventDetector()
    output = event_detector.forward("We believe it is essential to establish validated conflict-free sources of 3TG within the Democratic Republic of the Congo (the “DRC”) and adjoining countries (together, with the DRC, the “Covered Countries”), so that these minerals can be procured in a way that contributes to economic growth and development in the region. To aid in this effort, we have established a conflict minerals policy and an internal team to implement the policy.")
    print(output.event_type)