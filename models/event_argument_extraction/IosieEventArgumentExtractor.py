import os
from pathlib import Path
import gdown
from models.event_argument_extraction.EventArgumentExtractor import BaseEventArgumentExtractor
from schemes import EventArgumentExtractorOutput
from transformers import pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class IosieEventArgumentExtractor(BaseEventArgumentExtractor):
    def __init__(self, path_to_pretrained_model: str = "../data/event_argument_extractor/iosie"):
        super(IosieEventArgumentExtractor).__init__()
        self.prepare(path_to_pretrained_model)
        self.token_classifier = pipeline(
            "token-classification", model=path_to_pretrained_model, aggregation_strategy="simple")

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        result = self.token_classifier(tweet)
        tuples = self.get_tuples(result)
        entities = list(set([tuple[0] for tuple in tuples] + [tuple[-1] for tuple in tuples]))
        return EventArgumentExtractorOutput(tweet=tweet,
                                            event_arguments=entities,
                                            event_graph=tuples,
                                            wikidata_links={entity: None for entity in entities})

    @staticmethod
    def prepare(path_to_pretrained_model):
        if not Path(path_to_pretrained_model).exists():
            if not Path("../data/event_argument_extractor").exists():
                Path("../data/event_argument_extractor").mkdir()
            gdown.download_folder(
                url="https://drive.google.com/drive/folders/1nCbn6gb7Tq-8pEwSUeXBWHpMnOieAjfr?usp=sharing",
                output=path_to_pretrained_model)

    @staticmethod
    def get_tuples(predict_result):
        tuple = {"e1": "", 'predicate': "", 'tuples': []}
        for res in predict_result:
            if res["entity_group"] == 'P':
                tuple["predicate"] = res["word"]
            else:
                if res["entity_group"] == 'A0':
                    tuple["e1"] = res["word"]
                else:
                    tuple["tuples"].append([tuple["e1"], tuple["predicate"], res['word']])
        return tuple['tuples']