# DenseNet-169


This folder contains the script to run DenseNet-169 on ImageNet classification task. <br>
Variant supplied below in three different precisions accepts input of shape 224x224 and has 1.0x multiplier.

The original paper on the architecture is available here: https://arxiv.org/abs/1608.06993


### Accuracy:

|   | &nbsp;&nbsp;&nbsp;&nbsp; Top-1 Accuracy&nbsp;&nbsp;&nbsp;&nbsp;  |&nbsp;&nbsp;&nbsp;&nbsp; Top-5 Accuracy &nbsp;&nbsp;&nbsp;&nbsp; |
|:---:|:---:|:---:|
| FP32  | 74%  | 93 %  |
| INT8  | | |


### Performance on Intel(R) Xeon(R) Platinum 8124M CPU @ 3.00GHz

Latency:

|   | &nbsp;&nbsp;&nbsp;&nbsp; 1 thread&nbsp;&nbsp;&nbsp;&nbsp;  |&nbsp;&nbsp;&nbsp;&nbsp; 18 threads &nbsp;&nbsp;&nbsp;&nbsp; |
|:---:|:---:|:---:|
| FP32  | 64 ms | 16 ms  |
| INT8  | | |

Throughput (batch size = 16):

|   | &nbsp;&nbsp;&nbsp;&nbsp; 1 thread&nbsp;&nbsp;&nbsp;&nbsp;  |&nbsp;&nbsp;&nbsp;&nbsp; 18 threads &nbsp;&nbsp;&nbsp;&nbsp; |
|:---:|:---:|:---:|
| FP32  | 17.74 ips | 138.09 ips  |
| INT8  |           |             |


### Dataset and models

* validation dataset of 1000 images: <br>
  https://www.dropbox.com/s/nxgzz67tpux8wud/ILSVRC2012_onspecta.tar.gz  
  (note: you can unpack it using terminal command "tar -xf ./file")

* validation dataset labels for the reduced set: <br>
  https://www.dropbox.com/s/7ej242ym43v635i/imagenet_labels_onspecta.txt

* fp32 model: <br>
  https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/densenet169_fp32_pretrained_model.pb 
  

* fp16 model: <br>
  https://www.dropbox.com/s/ppzx4oz8ne9txeq/mobilenet_v2_fp16.pb


* int8 model: <br>
  https://www.dropbox.com/s/l2rnp6nnih69mop/densenet_169_quant.tflite 


### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the model zoo directory,
as well as DLS_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/model_zoo
export DLS_NUM_THREADS=1
```

For the best experience we also recommend setting environment variables as specified below.

```
export IMAGENET_IMG_PATH=/path/to/images
export IMAGENET_LABELS_PATH=/path/to/labels
```

Now you are able to run the run.py script. 

To get detailed information on the script's recognized arguments run it with -h flag for help.

The path to model (with a flag "-m") as well as its precision (with a flag "-p") have to be specified.\
Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.


Example command: 

```
python3 run.py -m /path/to/model.pb -p fp32
```
