# 碧落 #

毎日色々面白いことを試している。

### コードレシピ ###

* openpyxlでExcel編集

   会社でよく使われるExcelを自動で編集する。特に、同じ内容を複数の定型ファイルに更新するときに有効であろう。Excelはいいツールだが、各会社のフォーマットがそれぞれ違う。とはいえ、各社内部のフォーマットはある程度統一されたので、編集用スクリプトを作れば、何回も使われる。長期的に時間の節約はできるだろう。

```python
import openpyxl
wb = openpyxl.load_workbook(path)
ws = wb["Sheet1"]
ws["A1"].value = "X"
wb.save(path)
```

* OWASP ZAP 2.7でセキュリティ診断

   インジェクションなど脅威はシンプルで診断できる。
   サンプルは[ここ](https://github.com/zaproxy/zaproxy/wiki/ApiPython)
```python
import time
from zapv2 import ZAPv2
target = 'http://127.0.0.1:8080'
zap = ZAPv2(apikey=apikey)
scanid = zap.ascan.scan(target)
while (int(zap.ascan.status(scanid)) < 100):
    print 'Scan progress %: ' + zap.ascan.status(scanid)
    time.sleep(5)

```

* Seleniumよく使うメソッド

   ブラウザ自動操作で、退屈な事務作業を減らせよう。
~~~python
from selenium import webdriver
browser = webdriver.Ie() #他のブラウザも同様

browser.back()
browser.forward()
browser.refresh()
browser.current_url
browser.title
browser.page_source
browser.close()
browser.quit()

browser.find_element_by_class_name
browser.find_element_by_id
browser.find_element_by_xpath
element.click()
element.get_attribute("value")
~~~

* IP設定

   頻繁にIP切り替えが必要な時に有効
~~~dos
netsh interface ip set address [interface name] static [ip] [subnet mask] [gateway]

netsh interface ip set address "イーサネット" static 192.168.100.100 255.255.255.0 192.168.100.1
~~~

### ライブラリ インストール ###
~~~dos
# python 3.7
python -m pip install --upgrade pip
pip install numpy
pip install pandas
pip install requests
pip install selenium
pip install openpyxl
pip install pillow
pip install pyautogui
pip install python-owasp-zap-v2.4
pip install pywin32
pip install wmi
~~~

~~~dos
# python 3.6.5
pip install tensorflow
~~~