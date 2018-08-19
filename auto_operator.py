import os
from io import BytesIO
import xml.etree.ElementTree as et
from time import sleep

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.alert import Alert

from PIL import Image as pillowImage
import openpyxl
from openpyxl.drawing.image import Image as pyxlImage

base_path = r"C:\Users\linxu_000\Desktop\python script"
ieDriverPath = os.path.join(base_path,"bin","IEDriverServer.exe")
chromeDriverPath = os.path.join(base_path,"bin","chromedriver_win32","chromedriver.exe")
#browser = webdriver.Ie(ieDriverPath) #Windows10+IE11 required: https://github.com/SeleniumHQ/selenium/wiki/InternetExplorerDriver#required-configuration
browser = webdriver.Chrome(chromeDriverPath)

browser.get("http://localhost:8080")
sleep(2)

#driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#browser.execute_script("window.scrollTo(0, 500);")
#sleep(1)

linkElement = browser.find_element_by_xpath(r"./html/body/ul/li/p/a[@href='sqlijc']")
#linkElement.send_keys(Keys.CONTROL)
#sleep(0.5)
linkElement.click()
sleep(2)

usernameElement = browser.find_element_by_name("name")
usernameElement.send_keys("Mark")
browser.find_element_by_name("password").send_keys("password")

submitButton = browser.find_element_by_xpath(r"./html/body/form/input[@type='submit']")
submitButton.click()
sleep(2)

browser.back()

browser.get("http://localhost:8080")
sleep(2)

browser.find_element_by_link_text("XSS (クロスサイトスクリプティング)").click()
sleep(2)

browser.find_element_by_name("string").send_keys("abc")

browser.find_element_by_xpath(r"//input[@type='submit']").submit()
sleep(3)

browser.find_element_by_name("string").send_keys(r">tpircs/<;)eikooc.tnemucod(trela>tpIrcs<")

browser.find_element_by_xpath(r"//input[@type='submit']").click()
sleep(1)

Alert(browser).accept()

browser.get("http://localhost:8080")
sleep(2)

browser.find_element_by_link_text("サイズ制限の無いファイルアップロード").click()
sleep(1)

browser.find_element_by_name("file").send_keys(r"C:\Users\linxu_000\Desktop\python script\testFolder\testPic_klaudia.jpg")

browser.find_element_by_xpath(r"//input[@type='submit']").click()
sleep(1)

browser.back()