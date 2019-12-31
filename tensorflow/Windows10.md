# Windows10 tensorflow環境設定 #

## 環境 ##

1. OS
   Window 10 Home
   1903

1. GPU
   GeForce 1050
   Driver 441.66
   CUDA cuda_10.0.130_411.31_win10.exe

1. Python
   Python3.7.6

1. Tensorflow
   tensorflow2.0.0

## 設定手順 ##

1. Python

   [Download Python3.7.6](https://www.python.org/downloads/release/python-376/)

1. CUDA

   [Download CUDA Toolkit 10.0 Archive](https://developer.nvidia.com/cuda-10.0-download-archive)

1. TensorFlow

   ~~~powershell
   pip install tensorflow-gpu
   # python環境が複数ある場合、インストールフォルダ\Scriptsに遷移してから以下のコマンド実行
   .\pip install tensorflow-gpu
   ~~~

1. Optional Jupyter Notebook

   ~~~powershell
   cd C:\Users\linxu\AppData\Local\Programs\Python\Python37\Scripts # インストールパスのサンプル
   pip install jupyterlab
   pip install notebook
   jupyter notebook # Jupyter Notebook起動
   ~~~

## 環境確認 ##

1. コマンド

   ~~~powershell
   py -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
   # python環境が複数ある場合、インストールフォルダに遷移してから以下のコマンド実行
   .\python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

   # Output
   #2019-12-31 10:55:18.686448: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_100.dll
   #2019-12-31 10:55:21.091003: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
   #2019-12-31 10:55:22.082865: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
   #name: GeForce GTX 1050 major: 6 minor: 1 memoryClockRate(GHz): 1.493
   #pciBusID: 0000:01:00.0
   #2019-12-31 10:55:22.090677: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
   #2019-12-31 10:55:22.097597: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
   #2019-12-31 10:55:22.100903: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: #AVX2
   #2019-12-31 10:55:22.110128: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
   #name: GeForce GTX 1050 major: 6 minor: 1 memoryClockRate(GHz): 1.493
   #pciBusID: 0000:01:00.0
   #2019-12-31 10:55:22.117809: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
   #2019-12-31 10:55:22.124773: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
   #2019-12-31 10:55:25.049328: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
   #2019-12-31 10:55:25.055456: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0
   #2019-12-31 10:55:25.062023: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N
   #2019-12-31 10:55:25.067843: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3001 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050, pci bus id: 0000:01:00.0, compute capability: 6.1)
   #tf.Tensor(-1070.4426, shape=(), dtype=float32)
   ~~~
