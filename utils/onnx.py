import os
import time
import tensorflow as tf
import utils.benchmark as bench_utils
import onnxruntime as rt


class ONNXRunner:

    def __init__(self, path_to_model: str):
        self.__warm_up_run_latency = 0.0
        self.__total_inference_time = 0.0
        self.__times_invoked = 0
        self.__sess_options = self.__create_config(bench_utils.get_intra_op_parallelism_threads())
        print(self.__sess_options)
        print(type(self.__sess_options))

        self.__sess = rt.InferenceSession(path_to_model, sess_options=self.__sess_options)
        self.__feed_dict = dict()

    def __create_config(self, intra_threads: int):
        """
        A function creating config.

        :param intra_threads: int
        :param inter_threads: int
        :return: TensorFlow config
        """
        sess_options = rt.SessionOptions()
        sess_options.intra_op_num_threads = intra_threads
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        return sess_options

    def run(self, output_names):
        """
        A function running an input to onnx runner

        :param output_names:
        :param input:
        :return:
        """

        start = time.time()
        output_tensor = self.__sess.run(output_names, self.__feed_dict)
        finish = time.time()

        self.__total_inference_time += finish - start
        if self.__times_invoked == 0:
            self.__warm_up_run_latency += finish - start
        self.__times_invoked += 1

        return output_tensor

    def set_input_tensor(self, input_name: str, input_array):
        """
        A function assigning given numpy input array to the tensor under the provided input name.

        :param input_name: str, name of a input node in a model, eg. "image_tensor:0"
        :param input_array: numpy array with intended input
        """
        self.__feed_dict[input_name] = input_array

    def print_performance_metrics(self, batch_size):
        perf = bench_utils.print_performance_metrics(
            self.__warm_up_run_latency, self.__total_inference_time, self.__times_invoked, batch_size)
        return perf
