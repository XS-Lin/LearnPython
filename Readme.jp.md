# 碧落 #

毎日色々面白いことを試している。

## コードレシピ ##

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

### zap & selenium & openpyxl  ###

環境:

    Windows 10 64bit
    Excel 2013
    OWASP ZAP 2.7
    Selenium 3.14.0
    EasyBuggy 1.3.9
    IE11

設定:

[参照元(英語)](https://github.com/SeleniumHQ/selenium/wiki/InternetExplorerDriver#required-configuration)
    
    インターネット オプション

        セキュリティタブの「インターネット」「ローカル イントラネット」「信頼済みサイト」「制限付きサイト」の保護モードを同じ設定値にする。
        （全てオンまたはすべてオフ）
        
        詳細設定タブの「拡張保護モードを有効にする*」をオフにする
    
    Windows10 設定 -> ディスプレイ -> 拡大縮小とレイアウト -> 100%にする

    以下のレジストリキー追加(あれば不要)
    HKEY_LOCAL_MACHINE\SOFTWARE\Wow6432Node\Microsoft\Internet Explorer\Main\FeatureControl\FEATURE_BFCACHE
    サブキー iexplore.exe DWORD 0

~~~python
# seleinum で画面キャプチャーを撮って、Excelに保存
import os
from io import BytesIO
from selenium import webdriver
from PIL import Image as pillowImage
import openpyxl
from openpyxl.drawing.image import Image as pyxlImage

ieDriverPath = r"\IEDriverServer.exe" # IE Driver のパス
browser = webdriver.Ie(ieDriverPath)

browser.get("http://www.google.co.jp") # Googleを開いて、pythonを検索
textbox = browser.find_element_by_name('q')
textbox.send_keys('python')
textbox.submit()

screenShot = browser.get_screenshot_as_png() # 画面キャプチャーを撮る
im = pillowImage.open(BytesIO(screenShot))
img = pyxlImage(im)

wb = openpyxl.load_workbook(filename='test1.xlsx') # Excelテンプレート(空ファイル)
ws = wb["Sheet1"]
ws.add_image(img,'D3')
wb.save("test3.xlsx")
~~~

### ライブラリ インストール ###

~~~dos
# python 3.7
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
~~~

~~~dos
# python 3.6.5
pip install tensorflow
~~~

### その他 ###

* IP設定

   頻繁にIP切り替えが必要な時に有効

~~~dos
netsh interface ip set address [interface name] static [ip] [subnet mask] [gateway]

netsh interface ip set address "イーサネット" static 192.168.100.100 255.255.255.0 192.168.100.1
~~~