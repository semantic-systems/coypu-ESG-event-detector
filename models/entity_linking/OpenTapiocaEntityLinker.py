from typing import List
import spacy
from models.entity_linking.Linker import Linker


class OpenTapiocaEntityLinker(Linker):
    def __init__(self):
        super(OpenTapiocaEntityLinker, self).__init__()

    @staticmethod
    def instantiate():
        model = spacy.blank("en")
        model.add_pipe('opentapioca')
        return model

    def forward(self, text: str) -> List:
        model_output = self.model(text)
        return [(span.text, span.kb_id_, span.label_, span._.description) for span in model_output.ents]
