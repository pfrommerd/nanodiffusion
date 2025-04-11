from nanoconfig import config

import abc

@config
class ModelConfig(abc.ABC):
    @abc.abstractmethod
    def create(self, sample):
        """
        Create the model from the sample.
        :param sample: The sample to create the model from.
        :return: The model.
        """
        pass