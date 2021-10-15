import numpy as np
import pathlib
import utils.misc as utils


class RandomDataset:
    """
    A class providing facilities for preprocessing and postprocessing of ImageNet validation dataset.
    """

    def __init__(self):
        pass

    def get_input_array(self, target_shape):
        """
        A function returning an array containing pre-processed rescaled image's or multiple images' data.

        :param target_shape: tuple of intended image shape (height, width)
        :return: numpy array containing rescaled, pre-processed image data of batch size requested at class
        initialization
        """

        return np.random.rand(*target_shape)

    def submit_predictions(self, id_in_batch: int, byt):
        """
        A function meant for submitting a class predictions for a given image.

        :param id_in_batch: int, id of an image in the currently processed batch that the provided predictions relate to
        :param top_1_index: int, index of a prediction with highest confidence
        :param top_5_indices: list of ints, indices of 5 predictions with highest confidence
        :return:
        """
        pass

    def summarize_accuracy(self):
        """
        A function summarizing the accuracy achieved on the images obtained with get_input_array() calls on which
        predictions done where supplied with submit_predictions() function.
        """
        return {}
