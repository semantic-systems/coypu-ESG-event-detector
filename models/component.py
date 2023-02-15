

class BaseComponent(object):
    def __init__(self):
        self.version = self.__version__

    def forward(self, tweet: str):
        raise NotImplementedError

    @property
    def __version__(self):
        raise "1.0.0"
