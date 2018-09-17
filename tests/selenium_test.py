import os
from selenium import webdriver
import xml.etree.ElementTree as et
from PIL import Image as pillowImage
from io import BytesIO
import subprocess
from time import sleep

configFilePath = "config.xml"
configFilePath = r"D:\Site\MyScript\python_test\config_local.xml"
tree = et.ElementTree(file=configFilePath)
root = tree.getroot()
base_path_node = root.find(r"./settings/setting[@id='base_path']")
base_path = "" if base_path_node.text is None else base_path_node.text.strip()

temp_path_node = root.find(r"./settings/setting[@id='temp_folder']")
temp_path = "" if temp_path_node.text is None else temp_path_node.text.strip()

java_home_node = root.find(r"./settings/setting[@id='java_home']")
java_home = "" if java_home_node.text is None else java_home_node.text.strip()

#IE 11 64bit Add key:HKEY_LOCAL_MACHINE\SOFTWARE\Wow6432Node\Microsoft\Internet Explorer\Main\FeatureControl\FEATURE_BFCACHE    name=iexplore.exe  value=DWORD(0)
#for detail:https://github.com/SeleniumHQ/selenium/wiki/InternetExplorerDriver#required-configuration

#ieDriverPath = os.path.join(base_path,"bin","IEDriverServer.exe")
#browser = webdriver.Ie(ieDriverPath)
#browser.get("http://www.google.co.jp")
#textbox = browser.find_element_by_name('q')
#textbox.send_keys('python')
#textbox.submit()

#start test site "easybuggy". cmd: java -jar easybuggy.jar
#easybuggyPath = os.path.join(base_path,"bin","easybuggy.jar")
#javaPath = os.path.join(java_home,"bin","java.exe")
#cmd = javaPath + " -jar " + easybuggyPath
#subprocess.call(cmd)

ieDriverPath = os.path.join(base_path,"bin","IEDriverServer.exe")
browser = webdriver.Ie(ieDriverPath)
browser.get("http://localhost:8080")
sleep(1)

screenShot = browser.get_screenshot_as_png()
im = pillowImage.open(BytesIO(screenShot))
im.save(os.path.join(temp_path,"pic1.png"))

linkElement = browser.find_element_by_xpath(r"./html/body/ul/li/p/a[@href='sqlijc']")
linkElement.click()
linkElement.click()
sleep(1)

screenShot = browser.get_screenshot_as_png()
im = pillowImage.open(BytesIO(screenShot))
im.save(os.path.join(temp_path,"pic2.png"))

usernameElement = browser.find_element_by_name("name")
passwordElement = browser.find_element_by_name("password")
usernameElement.send_keys("Mark")
passwordElement.send_keys("password")

screenShot = browser.get_screenshot_as_png()
im = pillowImage.open(BytesIO(screenShot))
im.save(os.path.join(temp_path,"pic3.png"))

submitButton = browser.find_element_by_xpath(r"./html/body/form/input[@type='submit']")
submitButton.click()
sleep(1)

screenShot = browser.get_screenshot_as_png()
im = pillowImage.open(BytesIO(screenShot))
im.save(os.path.join(temp_path,"pic4.png"))

browser.close()
browser.quit()


#browser.back()
#browser.forward()
#browser.refresh()
#browser.current_url
#browser.title
#browser.page_source
#browser.close()
#browser.quit()

#browser.find_element_by_class_name
#browser.find_element_by_id
#browser.find_element_by_xpath
#element.click()

#actions = new Actions(browser)
#actions.move_to_element(element)
#actions.perform()

#element = browser.find_element_by_xpath("xpath")
#Select(element).select_by_index(indexnum) # indexで選択
#Select(element).select_by_value("value") # valueの値
#Select(element).select_by_text("text") # 表示テキスト

#element.send_keys("string")
#element.text

#element.get_attribute("value")
#Alert(browser).accept()

#browser.maximize_window()

#element.is_displayed()
#element.is_enabled()
#element.is_selected()