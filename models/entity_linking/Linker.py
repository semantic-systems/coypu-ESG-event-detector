from abc import abstractmethod


class Linker(object):
    def __init__(self):
        self.model = self.instantiate()

    @staticmethod
    @abstractmethod
    def instantiate():
        pass
         
    def forward(self, text: str):
        pass
