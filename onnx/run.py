import argparse
import os
import numpy as np
from utils.cv.imagenet import ImageNet
from utils.onnx import ONNXRunner
from utils.tflite import TFLiteRunner
from utils.benchmark import run_model
from utils.nlp.squad import Squad_v1_1
from transformers import AutoTokenizer
from utils.tokenization import FullTokenizer
from utils.run_onnx_squad import read_squad_examples, convert_examples_to_features


def parse_args():
    parser = argparse.ArgumentParser(description="Run Bert onnx")
    parser.add_argument("-m", "--model_path",
                        type=str, required=True,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--squad_path",
                        type=str, required=True,
                        help="path to directory with Squad 1.1 validation dataset")
    return parser.parse_args()


# preprocess input
predict_file = '/onspecta/Downloads/dev-v1.1.json'

# Use read_squad_examples method from run_onnx_squad to read the input file
eval_examples = read_squad_examples(input_file=predict_file)

max_seq_length = 256
doc_stride = 128
max_query_length = 64
batch_size = 1
n_best_size = 20
max_answer_length = 30

vocab_file = os.path.join('/onspecta', 'Downloads', 'vocab.txt')
tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=True)


def run_onnx_fp(model_path, batch_size, num_of_runs, timeout, squad_path):

    def run_single_pass(onnx_runner, squad):
        input_ids, input_mask, segment_ids, extra_data = convert_examples_to_features(eval_examples, tokenizer,
                                                                                      max_seq_length, doc_stride,
                                                                                      max_query_length)

        onnx_runner.set_input_tensor("input_ids:0", input_ids)
        onnx_runner.set_input_tensor("input_mask:0", input_mask)
        onnx_runner.set_input_tensor("segment_ids:0", segment_ids)
        onnx_runner.set_input_tensor("unique_ids_raw_output___9:0", np.random.randint(28, size=10))

        # tf_runner.set_input_tensor("input_ids:0", squad.get_input_ids_array())
        # tf_runner.set_input_tensor("input_mask:0", squad.get_attention_mask_array())
        # tf_runner.set_input_tensor("segment_ids:0", squad.get_token_type_ids_array())

        output = onnx_runner.run(['unstack:0', 'unstack:1'])

        # for i in range(batch_size):
        #     answer_start_id, answer_end_id = np.argmax(output["logits:0"][i], axis=0)
        #     squad.submit_prediction(
        #         i,
        #         squad.extract_answer(i, answer_start_id, answer_end_id)
        #     )

        # # preprocess input
        # predict_file = '/onspecta/Downloads/dev-v1.1.json'
        #
        # # Use read_squad_examples method from run_onnx_squad to read the input file
        # eval_examples = read_squad_examples(input_file=predict_file)
        #
        # max_seq_length = 256
        # doc_stride = 128
        # max_query_length = 64
        # batch_size = 1
        # n_best_size = 20
        # max_answer_length = 30
        #
        # vocab_file = os.path.join('uncased_L-12_H-768_A-12', 'vocab.txt')
        # tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
        #
        # # Use convert_examples_to_features method from run_onnx_squad to get parameters from the input
        # input_ids, input_mask, segment_ids, extra_data = convert_examples_to_features(eval_examples, tokenizer,
        #                                                                               max_seq_length, doc_stride,
        #                                                                               max_query_length)

    seq_size = 384
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    def tokenize(question, text):
        return tokenizer(question, text, add_special_tokens=True)

    def detokenize(answer):
        return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer))

    dataset = Squad_v1_1(batch_size, tokenize, detokenize, seq_size, squad_path)
    runner = ONNXRunner(model_path)

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def run_tf_fp32(model_path, batch_size, num_of_runs, timeout, images_path, labels_path):
    return run_tf_fp(model_path, batch_size, num_of_runs, timeout, images_path, labels_path)


def main():
    args = parse_args()
    if args.precision == "fp32":
        run_onnx_fp(
            args.model_path, args.batch_size, args.num_runs, args.timeout, args.squad_path
        )
        assert False


if __name__ == "__main__":
    main()
