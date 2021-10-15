import argparse

from utils.randomus import RandomDataset()
from utils.tf import TFFrozenModelRunner
from utils.benchmark import run_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3D Unet BRATs model.")
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
    return parser.parse_args()


def run_tf_fp32(model_path, batch_size, num_of_runs, timeout):

    def run_single_pass(tf_runner, randomus):
        shape = (batch_size, 4, 224, 224, 160)
        tf_runner.set_input_tensor("input:0", randomus.get_input_array(shape))
        output = tf_runner.run()
        for i in range(batch_size):
            randomus.submit_predictions(
                i,
                output["output:0"][i]
            )

    dataset = RandomDataset()
    runner = TFFrozenModelRunner(model_path, ["output:0"])
    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def main():
    args = parse_args()
    if args.precision == "fp32":
        run_tf_fp32(args.model_path, args.batch_size, args.num_runs, args.timeout)
    else:
        assert False


if __name__ == "__main__":
    main()
