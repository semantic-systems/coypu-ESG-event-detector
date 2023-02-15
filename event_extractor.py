import torch
from typing import List, Union, Optional

from models.event_argument_extraction.FalconEventArgumentExtractor import FalconEventArgumentExtractor
from models.event_argument_extraction.OpenTapiocaArgumentExtractor import OpenTapiocaArgumentExtractor
from models.event_argument_extraction.RebelEventArgumentExtractor import RebelEventArgumentExtractor
from models.event_detection.EventDetector import EventDetector
from models.event_detection.ESGEventDetector import ESGEventDetector
from schemes import EventExtractorOutput, EventDetectorOutput

EventDetectorType = Union[torch.nn.Module, EventDetector, ESGEventDetector]
EventArgumentExtractorType = Union[torch.nn.Module, FalconEventArgumentExtractor, RebelEventArgumentExtractor,
                                   OpenTapiocaArgumentExtractor]


class EventExtractor(object):
    def __init__(self,
                 event_detector: EventDetectorType,
                 event_argument_extractor: Optional[EventArgumentExtractorType] = None,
                 ):
        self.event_detector = event_detector
        self.event_argument_extractor = event_argument_extractor

    def extract_per_tweet(self, tweet: str) -> EventExtractorOutput:
        event_detector_output: EventDetectorOutput = self.event_detector.forward(tweet)
        return EventExtractorOutput(
            tweet=tweet,
            event_type=event_detector_output.event_type,
            wikidata_link=f"http://www.wikidata.org/entity/{event_detector_output.wikidata_link}")

    def extract_per_batch(self, tweets: List[str]) -> List[EventExtractorOutput]:
        pass

    def infer(self, tweet: str) -> tuple:
        output: EventExtractorOutput = self.extract_per_tweet(tweet)
        return output.event_type, output.wikidata_link


if __name__ == '__main__':
    event_detector = ESGEventDetector()
    event_extractor = EventExtractor(event_detector=event_detector)
    import gradio as gr

    title = "Hitec ESG Event Extraction Demo v1"
    description = "Event Detector: ESGBert (https://huggingface.co/nbroad/ESG-BERT)\n. Examples taken from Wikinews."
    article = '<img src="https://lh5.googleusercontent.com/5BFOPuJcVnIYF4P5bmvxTAYoD2cLsqFa_FZUyyUqYCivGq3XdfEPQJ_EvOMdwHD84957qvcitITbAMlKZi5E1nTBjrLXeG7rxPl2FbjJ_I8Ka7gnn7lX7ce1ZZUMlfzjn5R9hztf_q2K3T_lWMDki6AdtFuHT-YeikcG0j8QXCeWzONg6-t3e4NC4gmF" width=150px><img src="https://lh5.googleusercontent.com/vq_hxMF8lqmXdxIQJJGwq9SPRrE15SFRwCWHBWZXlFS_10LTAuzHmcx-1NUYznbXwH02KZgjpwz5eYM-j5m3RUOhW90rDV9uFj-30dBg7kS6irstL3_VHuHkOe5jskDM2-rTLAKhchDlrH3hAG8W9o9XIlzxWD-tQoj83oyrv6NVmsjhmAPR6ohDsmED" width=150px>'
    with gr.Blocks() as d:
        with gr.Row():
            input_box = gr.Textbox(placeholder="Enter a sentence here...", label="Input Tweet")
            output_box_event_type = gr.Textbox(label="Event type:")
            output_box_event_type_link = gr.Textbox(label="Event type link:")
    demo = gr.Interface(fn=event_extractor.infer,
                        inputs=input_box,
                        outputs= [output_box_event_type, output_box_event_type_link],
                        examples=[["\"One-in-100-year flood event\" devastates Western Australia"],
                                   ["118th United States Congress convenes; House of Representatives adjourns without electing Speaker for first time in 100 years."],
                                   ["UK Treasury considering plans for digital pound, economic secretary says."],
                                   ["Troops freed by Mali return to Ivory Coast."]
                                   ],
                        title=title,
                        description=description,
                        article=article)
    demo.launch(share=True, show_error=True)
