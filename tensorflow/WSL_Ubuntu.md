# Window10でLinux利用 #

現時点、WSLではCUDA使用不可、以下は調査過程

## ホスト環境 ##

1. ハードウェア
   |分類|名前|
   |---|---|---|
   | CPU |||
   | MEM |||
   | GPU |||

1. ソフトウェア
   |分類|名前|バージョン|
   |---|---|---|
   ||||

## WLS有効化 ##

   ~~~powershell
   # 管理者モードでPowshell起動
   # PS C:\WINDOWS\system32>
   Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
   # windows store で Ubuntu をインストール
   # 起動 -> linxs/plinxs
   ~~~

   ~~~bash
   sudo apt update
   sudo apt upgrade
   sudo apt install gcc

   # ----python3.6----
   sudo apt isntall python3-pip
   sudo pip3 install -U virtualenv
   cd ~
   virtualenv --system-site-packages -p python3 ./venv
   source ./venv/bin/activate　# 仮想環境開始
   # pip install --upgrade tensorflow #CPU版インストールは可能
   # pip install --upgrade tensorflow #GPU版はエラー
   pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-2.0.0-cp36-cp36m-manylinux2010_x86_64.whl
   python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))" # venv$
   pip install scikit-learn
   pip install pandas
   pip install matplotlib
   pip install seaborn
   deactivate # 仮想環境終了

   # TensorflowGPUサポート https://www.tensorflow.org/install/gpu
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
   sudo apt-get update
   wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
   sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
   sudo apt-get update
   sudo apt-get install --no-install-recommends nvidia-driver-418
   # Reboot. Check that GPUs are visible using the command: nvidia-smi
   # Install development and runtime libraries (~4GB)
   sudo apt-get install libnvidia-encode-440 nvidia-driver-440 xserver-xorg-video-nvidia-440
   sudo apt-get install cuda-drivers
   sudo apt-get install cuda-runtime-10-0 cuda-demo-suite-10-0
   sudo apt-get install --no-install-recommends cuda-10-0  libcudnn7=7.6.2.24-1+cuda10.0 libcudnn7-dev=7.6.2.24-1+cuda10.0
   sudo apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda10.0 libnvinfer-dev=5.1.5-1+cuda10.0
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

   # 以後、以下のコマンドで仮想環境開始、終了
   source ~/venv/bin/activate
   deactivate

   # 以下はtensorfolw環境と関係ない
   # ----python3.7 ----
   sudo apt install python3.7
   python3.7 -m pip install pip
   # python3.7 -m pip install <module>
   # ----sftp ---------
   /etc/init.d/ssh status
   sudo ssh-keygen -A
   sudo service ssh start
   ~~~
