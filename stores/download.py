import gdown
import argparse
from pathlib import Path
from dataclasses import dataclass


@dataclass
class LocalMetaConfig:
    name: str
    google_drive_link: str
    directory_to_store: str


def parse():
    parser = argparse.ArgumentParser(description='Please input the google drive link to download the model checkpoint.'
                                                 'As well as where the model should be stored. ')
    parser.add_argument("-name", "--name",
                        help='Name of the model to store.',
                        default="pretrain_model")
    parser.add_argument("-link", "--link",
                        help='Google Drive link to download the model checkpoint.',
                        default="https://drive.google.com/file/d/1AybsGHx6aSL4IP8NUWQsqMZaVvthQudd/view?usp=sharing")
    parser.add_argument("-path", "--path",
                        help='Path to a directory where the model checkpoint is saved.'
                             'This should NOT be located within this repo because we want to store big files within '
                             '/data',
                        default="./../../data")
    return parser.parse_args()


def download_from_google_drive(config: LocalMetaConfig):
    if not Path(config.directory_to_store).exists():
        Path(config.directory_to_store).mkdir(parents=True, exist_ok=False)
    if config.name.endswith(".pt"):
        gdown.download(config.google_drive_link,
                       str(Path(config.directory_to_store).joinpath(config.name).absolute()),
                       fuzzy=True)
    else:
        gdown.download_folder(url=config.google_drive_link,
                              output=str(Path(config.directory_to_store).joinpath(config.name).absolute()))


if __name__ == '__main__':

    event_detector_model_path = "../../../data/event_detector/crisisbert_w_oos_linear.pt"
    event_argument_extractor_model_path = "../../../data/event_argument_extractor/isoie"
    if not Path(event_detector_model_path).exists():
        if not Path("../../../data/event_detector").exists():
            Path("../../../data/event_detector").mkdir()
        gdown.download(url="https://drive.google.com/file/d/1Hj_s7UfKYOMszQYAYLy0iNFN1qD1wxrH/view?usp=sharing",
                       output=event_detector_model_path, fuzzy=True)
    if not Path(event_argument_extractor_model_path).exists():
        if not Path("../../../data/event_argument_extractor").exists():
            Path("../../../data/event_argument_extractor").mkdir()
        gdown.download_folder(
            url="https://drive.google.com/drive/folders/1nCbn6gb7Tq-8pEwSUeXBWHpMnOieAjfr?usp=sharing",
            output=event_argument_extractor_model_path)
    if not Path("../../../data/language_models/CoyPu-CrisisLM-v1").exists():
        if not Path("../../../data/language_models/").exists():
            Path("../../../data/language_models/").mkdir()
        gdown.download_folder(
            url="https://drive.google.com/drive/folders/1u6Mthkr4ffVNSjPn3F_B49axTsCHwRv8?usp=sharing",
            output="../../../data/language_models/CoyPu-CrisisLM-v1")
    # args = parse()
    # google_drive_link = args.link
    # directory_to_store = args.path
    # name = args.name
    # config = LocalMetaConfig(name=name, google_drive_link=google_drive_link, directory_to_store=directory_to_store)
    # download_from_google_drive(config)
