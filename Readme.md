# Install Lib #

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
