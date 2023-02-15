import json
import uuid
from datetime import datetime
from typing import Optional
from pathlib import Path
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import FOAF, RDF, RDFS
from schemes import EventDetectorOutput, EventArgumentExtractorOutput


class RDFGenerator(object):

    def convert(self,
                event_detector_output: EventDetectorOutput,
                event_argument_extractor_output: EventArgumentExtractorOutput,
                path_to_jsonld: Optional[str] = "../data/output.jsonld"):
        if not Path("../data").exists():
            Path("../data").mkdir()
        # first fix naming space
        coy = Namespace("https://schema.coypu.org/global#")

        g = Graph()
        g.bind("rdfs", RDFS)
        g.bind("rdf", RDF)
        g.bind("coy", coy)
        g.bind("foaf", FOAF)

        current = URIRef(f"https://data.coypu.org/event/mod/{uuid.uuid4()}")

        for linked_entity in event_argument_extractor_output.event_arguments:
            ID = URIRef("http://www.wikidata.org/entity/"+linked_entity.id)
            if linked_entity.label == "PERSON":
                # relations need to be found somewhere else
                g.add((current, coy.hasImpactOn, ID))
            if linked_entity.label == "LOC":
                g.add((current, coy.hasLocality, ID))
            if linked_entity.label == "ORG":
                g.add((current, coy.hasImpactOn, ID))
        # event type
        g.add((current, RDF.type, URIRef("https://schema.coypu.org/global#Event")))
        g.add((current, coy.hasEventType, URIRef("http://www.wikidata.org/entity/"+event_detector_output.wikidata_link)))
        g.add((current, RDFS.comment, Literal(event_detector_output.tweet)))
        g.add((current, coy.hasPublisher, Literal("HiTec")))
        g.add((current, coy.hasTimestamp, Literal(self.get_date_time())))
        g.serialize(format='json-ld', indent=4, destination=path_to_jsonld)
        return json.loads(g.serialize(format='json-ld', indent=4))[0]

    @staticmethod
    def get_date_time() -> str:
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        return dt_string
