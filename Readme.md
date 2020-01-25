# My Test Environment #

## Setting ##

### ハードウェア ###

1. 勉強用PC

    * CPU: Intel(R) Core(TM) i7-7700HQ

    * GPU: NVIDIA GeForce GTX 1050

    * MEMORY: 16 GB

2. 作業用PC

    * CPU: Intel(R) Core(TM) i7-8700K

    * GPU: NVIDIA GeForce GTX 1080 Ti

    * MEMORY: 64 GB

### ソフトウェア ###

1. CUDA Toolkit 10.0 (driver=v441.66)

2. Visual Studio 2019

3. Python

    各環境の動作確認と勉強のため、公式PythonとAnaconda両方を試します。

    * Python
      * Python 3.7.6
      * Python 3.8.1
    * Anaconda
      * Python 3.6.10

### Anaconda ###

1. mxnet用仮想環境

    ~~~powershell
    # environment.yml
    #    name: gluon
    #    dependencies:
    #    - python=3.6
    #    - pip:
    #      - mxnet-cu100==1.5.0
    #      - d2lzh==0.8.11
    #      - jupyter==1.0.0
    #      - matplotlib==2.2.2
    #      - pandas==0.23.4
    conda env create -f=env_name.yml
    ~~~

1. tensorflow用仮想環境

    ~~~powershell
    # TODO
    ~~~

### Python ###

ブラウザやローカルアプリを自動操作、画像処理、セキュリティー診断、データベース操作、Excelファイル自動処理、Web開発などモジュールを準備する。

~~~powershell
# python 3.7.6
python -m pip install --upgrade pip
pip install numpy
pip install pandas
pip install scipy
pip install requests
pip install selenium
pip install openpyxl
pip install pillow
pip install pyautogui
pip install pywin32
pip install matplotlib
pip install seaborn
pip install python-owasp-zap-v2.4
pip install django
pip install opencv-python
# pip install tensorflow
pip install tensorflow-gpu
pip install scikit-learn
pip install jupyterlab
pip install notebook
pip install mxnet-cu100
~~~

~~~powershell
# python 3.7.6
pip list --outdated
pip install --upgrade <package_name>
~~~

~~~powershell
# Upgrade all installed library by powshell for windows. (Tested on windows 10)
foreach ($line in @(pip list --outdated)) { if (-not($line.StartsWith("Package") -or $line.StartsWith("----"))) { pip install --upgrade $line.Split(" ")[0] } }
~~~
