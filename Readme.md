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

2. Visual Studio Code

3. Python

    各環境の動作確認と勉強のため、公式Python試します。

    * Python
      * Python 3.9.5

### Python ###

ブラウザやローカルアプリを自動操作、画像処理、セキュリティー診断、データベース操作、Excelファイル自動処理、Web開発などモジュールを準備する。

~~~powershell
# python 3.9.5
python -m pip install --upgrade pip
# ai 
pip install jupyterlab
pip install numpy scipy pandas matplotlib seaborn
# excel
pip install openpyxl
# broswer
pip install selenium
# auto
pip install pyautogui
# security
pip install python-owasp-zap-v2.4
# image
pip install opencv-python
# pip install tensorflow
pip install tensorflow-gpu
pip install scikit-learn
# web
pip install fastapi
pip install uvicorn[standard]
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
