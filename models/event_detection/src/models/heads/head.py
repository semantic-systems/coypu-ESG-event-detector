import abc

from torch.nn import Module


class Head(Module):
    def __init__(self):
        super(Head, self).__init__()

    @abc.abstractmethod
    def forward(self, **kwargs):
        raise NotImplementedError
