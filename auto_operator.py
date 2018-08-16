import os
from io import BytesIO
import xml.etree.ElementTree as et
from time import sleep

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

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
passwordElement = browser.find_element_by_name("password")
usernameElement.send_keys("Mark")
passwordElement.send_keys("password")

submitButton = browser.find_element_by_xpath(r"./html/body/form/input[@type='submit']")
submitButton.click()
sleep(2)
