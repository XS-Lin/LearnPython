# Python + Seleniumでブラウザ自動操作 #

本記事はPythonとSeleniumでブラウザ自動操作の方法を紹介します。

## 環境 ##

    Windows 10 64bit
    Excel 2013
    OWASP ZAP 2.7
    Selenium 3.14.0
    EasyBuggy 1.3.9
    IE11

設定について、この方の記事参照：

[SeleniumでInternet Explorer11を動かす方法](https://bitwave.showcase-tv.com/selenium%E3%81%A7internet-explorer11%E3%82%92%E5%8B%95%E3%81%8B%E3%81%99%E6%96%B9%E6%B3%95/)

## ライブラリ ##

~~~dos
# python 3.7
pip install selenium
pip install openpyxl
pip install pillow
pip install pyautogui
~~~

## 使い方 ##

以下、Seleniumを制御するサンプルを紹介します。

### 1.ライブラリインポート ###

~~~python
import os
from io import BytesIO
import xml.etree.ElementTree as et
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.support.ui import Select
from PIL import Image as pillowImage
import openpyxl
from openpyxl.drawing.image import Image as pyxlImage
~~~

### 2.Selenium起動 ###

~~~python
# パス設定、環境変数PATHに未設定の場合、設定が必要
ie_drive_path = r"" #IEDriverServer.exeのパス
chrome_drive_path = r"" #chromedriver.exeのパス

# IEDrive起動
# drive = webdriver.Ie(ieDriverPath)
# ChromeDrive起動
browser = webdriver.Chrome(chrome_drive_path)
~~~

### 3.ページを開く(GET) ###

~~~python
browser.get("http://localhost:8080")

# 連続操作の時に、ブラウザの実行時間を配慮する必要があります。
# 以後は割愛します。
time.sleep(1)

# ページ遷移確認
if browser.current_url = "http://localhost:8080/" :
    time.sleep(3)
~~~

### 4.要素取得 ###

以下の要素であれば、

~~~html
<input type="text" name="passwd" id="passwd-id" />
~~~

下記のいずれかで取得できます。ページに該当要素がない場合、NoSuchElementExceptionが発生します。

~~~python
element = browser.find_element_by_id("passwd-id")
element = browser.find_element_by_name("passwd")
element = browser.find_element_by_xpath("//input[@id='passwd-id']")
~~~

### 5.入力要素に値設定 ###

人の入力を模倣します。

~~~python
browser.get("http://localhost:8080/sqlijc")
browser.find_element_by_name("name").send_keys("Mark")
browser.find_element_by_name("password").send_keys("password")
~~~

要素「element」を対象にして、send_keysでキーボード操作を模倣できます。

~~~python
# テキストを入力
element.send_keys("some text")
# 矢印キー入力
element.send_keys(" and some", Keys.ARROW_DOWN)
# 入力内容クリア
element.clear()
# ドロップダウン取得
select = Select(driver.find_element_by_name('name'))
select = Select(driver.find_element_by_id('id'))
# ドロップダウン選択
select.select_by_index(index)
select.select_by_visible_text("text")
select.select_by_value(value)
# ドロップダウン選択をクリア
select.deselect_all()
~~~

### 6.送信(POST) ###

~~~python
# 画面を開いて、ユーザとパスワードを入力
browser.get("http://localhost:8080")
browser.find_element_by_name("name")send_keys("Mark")
find_element_by_name("password").send_keys("password")
# 送信ボタン取得
submitButton = browser.find_element_by_xpath(r"./html/body/form/input[@type='submit']")

# 送信ボタンクリック
submitButton.click()
~~~

### 7.ページスクロール ###

人の操作（フォーカス移動、ブラウザスクロール）を模倣します。

~~~python
browser.execute_script("window.scrollTo(0, 500);")
~~~

### 8.ダイアログ操作 ###

~~~python
# AlertのOKボタンクリック
Alert(browser).accept()

# ConfirmのOKボタンクリック
Alert(browser).accept()

# Confirmのキャンセルボタンクリック
Alert(browser).dismiss()
~~~

### 9.ブラウザ終了 ###

~~~python
browser.close()
~~~

## ソース例 ##

バグ勉強用サイト「EasyBuggy 1.3.9」を利用します。

起動コマンド

~~~dos
java -jar easybuggy.jar
~~~

起動サイト

~~~url
http://localhost:8080
~~~

フォルダ構成

~~~path
baseFolder
 |-bin
 |  |-IEDriverServer.exe
 |  |-chromedriver_win32
 |     |-chromedriver.exe
 |
 |-testFolder
    |-testPic_klaudia.jpg
~~~

サンプルソース

~~~python
import os
from io import BytesIO
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.alert import Alert
from PIL import Image as pillowImage
import openpyxl
from openpyxl.drawing.image import Image as pyxlImage

base_path = r"" #baseFolderパス
ieDriverPath = os.path.join(base_path,"bin","IEDriverServer.exe")
chromeDriverPath = os.path.join(base_path,"bin","chromedriver_win32","chromedriver.exe")
browser = webdriver.Chrome(chromeDriverPath)

browser.get("http://localhost:8080")
time.sleep(2)

browser.current_url

linkElement = browser.find_element_by_xpath(r"./html/body/ul/li/p/a[@href='sqlijc']")
linkElement.click()
time.sleep(2)

usernameElement = browser.find_element_by_name("name")
usernameElement.send_keys("Mark")
browser.find_element_by_name("password").send_keys("password")

submitButton = browser.find_element_by_xpath(r"./html/body/form/input[@type='submit']")
submitButton.click()
time.sleep(2)

browser.back()

browser.get("http://localhost:8080")
time.sleep(2)

browser.find_element_by_link_text("XSS (クロスサイトスクリプティング)").click()
time.sleep(2)

browser.find_element_by_name("string").send_keys("abc")

browser.find_element_by_xpath(r"//input[@type='submit']").submit()
time.sleep(3)

browser.find_element_by_name("string").send_keys(r">tpircs/<;)eikooc.tnemucod(trela>tpIrcs<")

browser.find_element_by_xpath(r"//input[@type='submit']").click()
time.sleep(1)

Alert(browser).accept()

browser.get("http://localhost:8080")
time.sleep(2)

browser.find_element_by_link_text("サイズ制限の無いファイルアップロード").click()
time.sleep(1)

browser.find_element_by_name("file").send_keys(base_path + r"\testFolder\testPic_klaudia.jpg")

browser.find_element_by_xpath(r"//input[@type='submit']").click()
time.sleep(1)

browser.back()
browser.close()
browser.quit()
~~~

## 参考資料 ##

[1. Selenium Download](https://www.seleniumhq.org/download/)

[2. Selenium with Python](https://selenium-python.readthedocs.io/)