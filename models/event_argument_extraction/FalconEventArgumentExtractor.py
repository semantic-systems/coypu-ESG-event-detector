import requests
from models.event_argument_extraction.EventArgumentExtractor import BaseEventArgumentExtractor
from schemes import EventArgumentExtractorOutput


class FalconEventArgumentExtractor(BaseEventArgumentExtractor):
    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        json_data = {
            'text': tweet,
        }
        headers = {
            'Content-Type': 'application/json',
        }
        response = requests.post('https://labs.tib.eu/falcon/falcon2/api?mode=long', json=json_data, headers=headers)
        entities = response.json().get("entities_wikidata", None)
        relations = response.json().get("relations_wikidata", None)
        if entities is not None:
            entity = [entity["surface form"] for entity in entities] + [relation["surface form"] for relation in relations]
            wikidata_links = [entity["URI"] for entity in entities] + [relation["URI"] for relation in relations]
            return EventArgumentExtractorOutput(tweet=tweet,
                                                event_arguments=entity,
                                                event_graph=None,
                                                wikidata_links={e: wikidata_links[i] for i, e in enumerate(entity)})
        else:
            return EventArgumentExtractorOutput(tweet=tweet,
                                                event_arguments=None,
                                                event_graph=None,
                                                wikidata_links=None)

