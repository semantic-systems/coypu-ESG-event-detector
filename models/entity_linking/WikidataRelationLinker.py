from typing import Dict, List
import json, urllib
import urllib.request
from models.entity_linking.Linker import Linker


class WikidataRelationLinker(Linker):
    def __init__(self):
        super(WikidataRelationLinker, self).__init__()

    @staticmethod
    def instantiate() -> Dict:
        pass

    def forward(self, text: str) -> list:
        return self.filter(self.get_related_property_entities(text))

    @staticmethod
    def get_id_from_mention(entity_name: str) -> str:
        # Get wikidata id from wikidata api
        ans = []
        url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&search=" + "+".join(
            entity_name.split(" ")) + "&language=en"
        response = json.loads(urllib.request.urlopen(url).read())
        ans += response["search"]
        if (ans == [] and " " in entity_name):
            # Reverse Trick : Pan Changjiang
            url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&search=" + "+".join(
                entity_name.split(" ")[::-1]) + "&language=en"
            response = json.loads(urllib.request.urlopen(url).read())
            ans += response["search"]
        if (ans == [] and len(entity_name.split(" ")) > 2):
            # Abbreviation Trick
            url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&search=" + "+".join(
                [entity_name.split(" ")[0], entity_name.split(" ")[-1]]) + "&language=en"
            response = json.loads(urllib.request.urlopen(url).read())
            ans += response["search"]
        if len(ans) > 0:
            # Returns the first one, most likely one
            return ans[0]["id"]
        else:
            # Some outliers : Salvador Domingo Felipe Jacinto Dali i Domenech - Q5577
            return "Not Applicable"

    @staticmethod
    def get_entity_name_from_id(entity_id: str) -> str:
        # Get entity name given entity id
        url = "https://www.wikidata.org/w/api.php?action=wbgetentities&props=labels&ids=" + entity_id + "&languages=en&format=json"
        response = json.loads(urllib.request.urlopen(url).read())
        return response["entities"][entity_id]["labels"]["en"]["value"]

    def get_related_property_entities(self, name: str) -> List:
        # Get related property-entity (property id, property name, entity id, entity name) given entity name
        # Return a list of dicts, each dict contains (pid, property, eid, entity)
        # Fail to fetch eid would result in empty list
        query = self.get_id_from_mention(name)
        if query == "Not Applicable": return []
        ans = []
        url = "https://www.wikidata.org/w/api.php?action=wbgetentities&ids=" + query + "&format=json&languages=en"
        response = json.loads(urllib.request.urlopen(url).read())
        for p in response["entities"][query]["claims"]:
            for c in response["entities"][query]["claims"][p]:
                # Enumerate property & entity (multi-property, multi-entity)
                try:
                    # Some properties are not related to entities, thus try & except
                    cid = c["mainsnak"]["datavalue"]["value"]["id"]
                    ans.append({
                        "pid": p,
                        "entity": self.get_entity_name_from_id(cid)
                    })
                except:
                    continue
        return ans

    @staticmethod
    def filter(ans: List, choose='P') -> List:
        return [{'ID': ent['pid'], 'label': ent['entity']}
                for ent in ans if ent['pid'].startswith(choose)]


if __name__ == "__main__":
    test = WikidataRelationLinker()
    print(test.forward('border'))