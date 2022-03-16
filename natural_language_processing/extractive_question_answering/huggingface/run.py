import argparse

import numpy as np
import tensorflow as tf
import torch

from utils.benchmark import run_model
from utils.nlp.squad import Squad_v1_1
from utils.pytorch import PyTorchRunner
from utils.tf import TFSavedModelRunner
from utils.misc import print_goodbye_message_and_die
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering, AutoModelForQuestionAnswering


def parse_args():
    parser = argparse.ArgumentParser(description="Run model from Hugging Face's transformers repo for "
                                                 "extractive question answering task.")
    parser.add_argument("-m", "--model_name",
                        type=str, default="bert-large-uncased-whole-word-masking-finetuned-squad",
                        help="name of the model")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str, default="tf",
                        choices=["tf", "pytorch"],
                        help="specify the framework in which a model should be run")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--squad_path",
                        type=str,
                        help="path to directory with SQuAD 1.1 dataset")
    return parser.parse_args()


def run_tf(model_name, batch_size, num_runs, timeout, squad_path, **kwargs):

    def run_single_pass(tf_runner, squad):

        output = tf_runner.run(np.array(squad.get_input_ids_array(), dtype=np.int32))

        for i in range(batch_size):
            answer_start_id = np.argmax(output.start_logits[i])
            answer_end_id = np.argmax(output.end_logits[i])

            print(answer_start_id)
            print(type(answer_start_id))
            print(answer_end_id)
            print(type(answer_end_id))

            quit()
            squad.submit_prediction(
                i,
                squad.extract_answer(i, answer_start_id, answer_end_id)
            )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(question, text):
        return tokenizer(question, text, add_special_tokens=True)

    def detokenize(answer):
        return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer))

    dataset = Squad_v1_1(batch_size, tokenize, detokenize, dataset_path=squad_path)
    runner = TFSavedModelRunner()
    runner.model = tf.function(TFAutoModelForQuestionAnswering.from_pretrained(model_name))

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch(model_name, batch_size, num_runs, timeout, squad_path, **kwargs):

    def run_single_pass(pytorch_runner, squad):

        # print(squad.get_input_ids_array())
        # print(type(squad.get_input_ids_array()))
        # quit()
        #
        # print(torch.from_numpy(squad.get_input_ids_array()).type(torch.int32))
        # print(type(torch.from_numpy(squad.get_input_ids_array()).type(torch.int32)))
        # quit()

        output = pytorch_runner.run(torch.from_numpy(squad.get_input_ids_array()).type(torch.int32))

        for i in range(batch_size):
            print(output.start_logits[i])
            print(type(output.start_logits[i]))
            print(output.end_logits[i])
            print(type(output.end_logits[i]))

            # quit()
            answer_start_id = np.argmax(output.start_logits[i]).item()
            answer_end_id = np.argmax(output.end_logits[i]).item()

            print(answer_start_id)
            print(type(answer_start_id))
            print(answer_end_id)
            print(type(answer_end_id))

            quit()
            squad.submit_prediction(
                i,
                squad.extract_answer(i, answer_start_id, answer_end_id)
            )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(question, text):
        return tokenizer(question, text, add_special_tokens=True)

    def detokenize(answer):
        return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer))

    dataset = Squad_v1_1(batch_size, tokenize, detokenize, dataset_path=squad_path)
    runner = PyTorchRunner(AutoModelForQuestionAnswering.from_pretrained(model_name), disable_jit_freeze=True)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def main():
    args = parse_args()
    if args.framework == "tf":
        run_tf(**vars(args))
    elif args.framework == "pytorch":
        run_pytorch(**vars(args))
    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
