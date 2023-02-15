import spacy
from models.event_argument_extraction.EventArgumentExtractor import BaseEventArgumentExtractor
from schemes import EventArgumentExtractorOutput, LinkedEntity


class OpenTapiocaArgumentExtractor(BaseEventArgumentExtractor):
    def __init__(self):
        super(OpenTapiocaArgumentExtractor, self).__init__()
        self.model = self.instantiate()
        self.blacklist = ["Q48"]

    @staticmethod
    def instantiate():
        model = spacy.blank("en")
        model.add_pipe('opentapioca')
        return model

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        model_output = self.model(tweet)
        linked_entities = [LinkedEntity(entity=entity.text,
                                        id=entity.kb_id_,
                                        label=entity.label_,
                                        description=entity._.description) for entity in model_output.ents if entity.kb_id_ not in self.blacklist]
        return EventArgumentExtractorOutput(tweet=tweet,
                                            event_arguments=linked_entities)
