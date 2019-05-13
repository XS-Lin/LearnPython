# 碧落 #

毎日色々面白いことを試している。

## ライブラリ インストール ##

~~~dos
# python 3.7.3
python -m pip install --upgrade pip
pip install numpy
pip install pandas
pip install scipy
pip install requests
pip install selenium
pip install openpyxl
pip install pillow
pip install pyautogui
pip install python-owasp-zap-v2.4
pip install pywin32
pip install matplotlib
pip install seaborn
pip install django
pip install tensorflow
~~~

~~~dos
# python 3.7.3
pip list --outdated
pip install --upgrade <package_name>
~~~

### その他 ###

* IP設定

   頻繁にIP切り替えが必要な時に有効

~~~dos
netsh interface ip set address [interface name] static [ip] [subnet mask] [gateway]

netsh interface ip set address "イーサネット" static 192.168.100.100 255.255.255.0 192.168.100.1
~~~