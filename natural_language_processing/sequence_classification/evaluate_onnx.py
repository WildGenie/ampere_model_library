import os
import psutil

# ATTENTION: these environment variables must be set before importing onnxruntime.
os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=True))
os.environ["OMP_WAIT_POLICY"] = 'ACTIVE'


import nlp_utils as nu
import onnxruntime
from tqdm import tqdm
import argparse
import numpy as np
import time

# https://huggingface.co/transformers/v3.0.2/training.html#tensorflow

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Hugging Face models for GLUE tasks dataset")
    parser.add_argument("-m", "--model_name",
                        type=str, required=True,
                        help="name of the pre-trained model from which the \
                        trained model was trained")
    parser.add_argument("-p", "--model_path",
                        type=str, required=True,
                        help="path of the trained model")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=8,
                        help="batch size to feed the model with")
    parser.add_argument("-t", "--task_name",
                        type=str, default="mrpc",
                        help="Task name of the GLUE dataset")
    return parser.parse_args()


def evaluate_accuracy(model_path, is_model_bert, eval_dataset):

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])

    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataset):
        # print("Input batch ", batch)
        # batch = tuple(t.detach().cpu().numpy() for t in batch)
        if(is_model_bert):
            ort_inputs = {
                            'input_ids':  batch[0]['input_ids'].numpy(),
                            'attention_mask': batch[0]['attention_mask'].numpy(),
                            'token_type_ids': batch[0]['token_type_ids'].numpy(),
                        }
        else:
            ort_inputs = {
                            'input_ids':  batch[0]['input_ids'].numpy(),
                            'attention_mask': batch[0]['attention_mask'].numpy(),
                            # 'token_type_ids': batch[0]['token_type_ids'].numpy(),
                        }
        logits = np.reshape(session.run(None, ort_inputs), (-1,2))
        if preds is None:
            preds = logits
        else:
            preds = np.append(preds, logits, axis=0)

    # Measure the latency.
    total_runs = 100
    num_tokens = 128
    start = time.time()
    for _ in range(total_runs):
        results = session.run(None, ort_inputs)
    end = time.time()
    print("ONNX Runtime cpu inference time for sequence length {}: {} ms".format(num_tokens, format((end - start) * 1000 / total_runs, '.2f')))

    return preds

def evaluate_latency(model_path):

    sess_options = onnxruntime.SessionOptions()

    # intra_op_num_threads=1 can be used to enable OpenMP in OnnxRuntime 1.2.0.
    # For OnnxRuntime 1.3.0 or later, this does not have effect unless you are using onnxruntime-gpu package.
    # sess_options.intra_op_num_threads=1

    # Providers is optional. Only needed when you use onnxruntime-gpu for CPU inference.
    session = onnxruntime.InferenceSession(output_model_path, sess_options, providers=['CPUExecutionProvider'])

    batch_size = 1
    inputs_onnx = {k_: numpy.repeat(v_, batch_size, axis=0) for k_, v_ in inputs.items()}

    # Warm up with one run.
    results = session.run(None, inputs_onnx)

    # Measure the latency.
    start = time.time()
    for _ in range(total_runs):
        results = session.run(None, inputs_onnx)
    end = time.time()
    print("ONNX Runtime cpu inference time for sequence length {} (model not optimized): {} ms".format(num_tokens, format((end - start) * 1000 / total_runs, '.2f')))
    del session

def main():
    args = parse_args()    
    is_model_bert = True
    if("distil" in args.model_name):
        is_model_bert = False
    eval_dataset, eval_label = nu.get_dataset(args.model_name, args.task_name, 
        args.batch_size, is_model_bert)
    preds = evaluate_accuracy(args.model_path, is_model_bert, eval_dataset)
    result = nu.compute_metrics(preds, eval_label, args.task_name)
    print("Evaluation results ", result)

    # evaluate_latency(args.model_path)



main()



