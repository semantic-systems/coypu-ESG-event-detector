from pathlib import Path

import pytest

from models.event_detection import EventDetector
from event_extractor import EventExtractor


@pytest.fixture()
def pretrained_model_google_drive_url():
    return "https://drive.google.com/file/d/1AybsGHx6aSL4IP8NUWQsqMZaVvthQudd/view?usp=sharing"


@pytest.fixture()
def pretrained_model_output_path(tmpdir):
    return tmpdir.mkdir("outputs")


@pytest.fixture()
def downloaded_pretrained_model_path(pretrained_model_output_path):
    return str(pretrained_model_output_path)+"/mocked_checkpoint.pt"


@pytest.fixture()
def valid_pretrained_event_detector_path():
    return str(Path(__file__).parent.parent.joinpath("../data/pretrained_event_detector.pt"))


@pytest.fixture()
def invalid_pretrained_event_detector_path(pretrained_model_output_path):
    return str(Path(pretrained_model_output_path).joinpath("mocked_checkpoint.pt"))


@pytest.fixture()
def valid_pretrained_event_argument_extractor_path():
    return str(Path(__file__).parent.parent.joinpath("../data/pretrained_event_argument_extractor.pt"))


@pytest.fixture()
def invalid_pretrained_event_argument_extractor_path(pretrained_model_output_path):
    return str(Path(pretrained_model_output_path).joinpath("mocked_checkpoint.pt"))


@pytest.fixture()
def event_type_detector_instance(valid_pretrained_event_detector_path):
    return EventDetector(valid_pretrained_event_detector_path)


@pytest.fixture()
def event_extractor_instance(event_type_detector_instance, event_argument_extractor_openie_instance):
    return EventExtractor(event_type_detector_instance, event_argument_extractor_openie_instance)
