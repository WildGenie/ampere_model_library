from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments
import argparse
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark NLP models")
    parser.add_argument("-m", "--model", type=str, default="bert-base-uncased", required=False,
                        help="you can see the available list of models here: "
                        "https://huggingface.co/transformers/pretrained_models.html")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=8, required=False,
                        help="batch size to feed the model with")
    parser.add_argument("--sequence_length", required=False,
                        type=int, default=8,
                        help="sequence length to feed the model with")
    parser.add_argument("--intra", required=False,
                        type=int, default=1,
                        help="set the intra threads for TF")
    return parser.parse_args()


def benchmark_nlp_model(model, batch_size, sequence_length, intra):

    tf.config.threading.set_intra_op_parallelism_threads(intra)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    args = TensorFlowBenchmarkArguments(models=[model], batch_sizes=[batch_size],
                                        sequence_lengths=[sequence_length], memory=True)

    benchmark = TensorFlowBenchmark(args)
    results = benchmark.run()

    print(results)


def main():
    args = parse_args()
    benchmark_nlp_model(args.model, args.batch_size, args.sequence_length, args.intra)


if __name__ == "__main__":
    main()
    
