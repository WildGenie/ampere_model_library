import argparse
import torch
import sentencepiece
from utils.benchmark import run_model
from utils.misc import print_goodbye_message_and_die


def parse_args():
    parser = argparse.ArgumentParser(description="Run en-de model.")
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], default="fp32",
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str,
                        choices=["pytorch"], default="pytorch",
                        help="specify the framework in which a model should be run")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--tokenizer_path",
                        type=str,
                        help="path to the tokenizer model")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--disable_jit_freeze", action='store_true',
                        help="if true model will be run not in jit freeze mode")
    return parser.parse_args()


class Dataset:
    def __init__(self):
        pass

    def summarize_accuracy(self):
        pass

    def reset(self):
        pass


def run_pytorch_fp(model_path, batch_size, num_runs, timeout, tokenizer_path, disable_jit_freeze=False):
    from utils.pytorch import PyTorchRunner

    def run_single_pass(pytorch_runner, template_dataset):
        tensor = tokenizer.encode("aaaaaaaaa", out_type=str)
        output = pytorch_runner.run(tensor)
        print(output)

    tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
    runner = PyTorchRunner(torch.load(model_path), disable_jit_freeze=disable_jit_freeze)
    dataset = Dataset()

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch_fp32(model_path, batch_size, num_runs, timeout, tokenizer_path, **kwargs):
    return run_pytorch_fp(model_path, batch_size, num_runs, timeout, tokenizer_path)


def main():
    args = parse_args()

    if args.framework == "pytorch":
        if args.precision == "fp32":
            run_pytorch_fp32(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)
    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
