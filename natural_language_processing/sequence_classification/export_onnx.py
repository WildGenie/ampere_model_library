import onnxruntime
import tf2onnx
# optimize transformer-based models with onnxruntime-tools
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions
from transformers import TFBertForSequenceClassification, TFDistilBertForSequenceClassification
import tensorflow as tf
import time
import argparse
import nlp_utils as nu
import os

# https://huggingface.co/transformers/v3.0.2/training.html#tensorflow

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Hugging Face models for GLUE tasks dataset")
    parser.add_argument("-m", "--model_name",
                        type=str, required=True,
                        help="name of the pre-trained model from which the \
                        trained model was trained")
    # parser.add_argument("-p", "--model_path",
    #                     type=str, required=True,
    #                     help="path of the trained model")
    parser.add_argument("-o", "--output_path",
                        type=str, required=True,
                        help="path of the converted onnx model")
    parser.add_argument("-t", "--task_name",
                        type=str, default="mrpc",
                        help="Task name of the GLUE dataset")
    parser.add_argument("-op", "--optimize",
                        type=bool, default=True,
                        help="Optimize the model")
    parser.add_argument("-q", "--quant_int8",
                        type=bool, default=True,
                        help="Quantize model to int8")
    return parser.parse_args()



def export_onnx_model(model, onnx_model_path, is_model_bert):

    # describe the inputs
    if(is_model_bert):
        input_spec = (
            tf.TensorSpec((None,  None), tf.int32, name="input_ids"),
            tf.TensorSpec((None,  None), tf.int32, name="token_type_ids"),
            tf.TensorSpec((None,  None), tf.int32, name="attention_mask")
        )
    else:
        input_spec = (
            tf.TensorSpec((None,  None), tf.int32, name="input_ids"),
            tf.TensorSpec((None,  None), tf.int32, name="attention_mask")
        )
    start = time.time()

    _, _ = tf2onnx.convert.from_keras(model, 
        input_signature=input_spec, opset=13, 
        output_path=onnx_model_path)



def optimize_onnx_model(unopt_model_path, opt_model_path):
    # disable embedding layer norm optimization for better model size reduction
    opt_options = BertOptimizationOptions('bert')
    opt_options.enable_embed_layer_norm = False

    opt_model = optimizer.optimize_model(
        unopt_model_path,
        'bert', 
        num_heads=12,
        hidden_size=768,
        # model_type='tf2onnx',
        optimization_options=opt_options)
    opt_model.save_model_to_file(opt_model_path)
    return


def quantize_onnx_model(onnx_model_path, quantized_model_path):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnx
    onnx_opt_model = onnx.load(onnx_model_path)
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QInt8)
    return

def main():
    args = parse_args()    
    is_model_bert = True
    if("distil" in args.model_name):
        is_model_bert = False

    model_path = nu.get_model(is_model_bert)

    # Load pre-trained models
    if(is_model_bert):
        model = TFBertForSequenceClassification.from_pretrained(model_path)
    else:
        model = TFDistilBertForSequenceClassification.from_pretrained(model_path)

    export_onnx_model(model, args.output_path, is_model_bert)
    output_basename = args.output_path[:-5]
    if(args.optimize):
        optimize_onnx_model(args.output_path, output_basename+".opt.onnx")
    if(args.quant_int8):
        quantize_onnx_model(output_basename+".opt.onnx", 
            output_basename+".opt.quant.onnx")

        print('ONNX full precision model size (MB):', 
            os.path.getsize(output_basename+".opt.onnx")/(1024*1024))
        print('ONNX quantized model size (MB):', 
            os.path.getsize(output_basename+".opt.quant.onnx")/(1024*1024))

main()
