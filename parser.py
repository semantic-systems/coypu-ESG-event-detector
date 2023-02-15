import argparse


def parse():
    parser = argparse.ArgumentParser(description='Please select a configuration in ./config')

    parser.add_argument("-c", "--config",
                        help='Path to a configuration file in yaml format.',
                        default="./config/version_two.yml")
    parser.parse_args()
    return parser.parse_args()