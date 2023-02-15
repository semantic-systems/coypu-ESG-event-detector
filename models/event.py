import json
from typing import List, Union

from schemes import Time, Location


class Event(object):
    id: str = None
    event_type: str = None
    event_start_time: Time = None
    event_end_time: Time = None
    event_focus_time: Time = None
    event_location: Location = None
    severity: float = None
    num_tweets: int = None
    tweets_store: str = None
    event_arguments: List = None
    event_relation: List = None
    wikidata_links: List = None
    creation_time: str = None


class EventStore(object):
    def __init__(self):
        self.events: Union[List[Event], None] = None

    def add_event(self, event: Event) -> None:
        self.events.append(event)

    def remove_event_by_id(self, event_id: str) -> None:
        self.events.pop(self.get_event_index_by_id(event_id))

    def remove_event_by_name(self, name: str) -> None:
        self.events.pop(self.get_event_index_by_name(name))

    def get_event_index_by_id(self, event_id: str) -> int:
        event = list(filter(lambda event: event.id == event_id, self.events))[0]
        return self.events.index(event)

    def get_event_index_by_name(self, name: str) -> int:
        event = list(filter(lambda event: event.event_type == name, self.events))[0]
        return self.events.index(event)

    def to_json(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.events, f, ensure_ascii=False, indent=4)
