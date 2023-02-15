import os
from typing import Dict

from models.entity_linking.OpenTapiocaEntityLinker import OpenTapiocaEntityLinker
from models.entity_linking.WikidataRelationLinker import WikidataRelationLinker
from models.event_argument_extraction.EventArgumentExtractor import BaseEventArgumentExtractor
from models.knowledge_extraction.RebelKnowledgeExtractor import RebelKnowledgeExtractor
from schemes import EventArgumentExtractorOutput

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RebelEventArgumentExtractor(BaseEventArgumentExtractor):
    def __init__(self, path_to_pretrained_model: str = "useless_path"):
        super(RebelEventArgumentExtractor).__init__()
        self.model = RebelKnowledgeExtractor()
        self.entity_linker = OpenTapiocaEntityLinker()
        self.relation_linker = WikidataRelationLinker()

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        results = self.model.forward(tweet)
        entities = list(set([result["head"] for result in results]+[result["tail"] for result in results]))
        relations = list(set([result["relation"] for result in results]))
        event_graph = [[result["head"], result["relation"], result["tail"]] for result in results]
        wikidata_links = {}
        for entity in entities:
            wikidata_links.update(self.get_entity_link(entity))
        for relation in relations:
            wikidata_links.update(self.get_relation_link(relation))
        return EventArgumentExtractorOutput(tweet=tweet,
                                            event_arguments=entities,
                                            event_graph=event_graph,
                                            wikidata_links=wikidata_links)

    def get_entity_link(self, mention: str) -> Dict:
        link = self.entity_linker.forward(mention)
        if link:
            return {mention: "http://www.wikidata.org/entity/" + link[0][1]}
        else:
            return {}

    def get_relation_link(self, relation: str) -> Dict:
        link = self.relation_linker.forward(relation)
        if link:
            return {relation: "http://www.wikidata.org/entity/" + link[0]["ID"]}
        else:
            return {}