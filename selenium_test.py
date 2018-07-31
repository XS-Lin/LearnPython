import os
from selenium import webdriver
import xml.etree.ElementTree as et

base_path = ""
configFilePath = "config.xml"
#base_path=r"C:\Users\linxu_000\Desktop\python script"
#configFilePath=r"C:\Users\linxu_000\Desktop\python script\config_local.xml"
#base_path=r"D:\Site\MyScript\python_test"
#configFilePath=r"D:\Site\MyScript\python_test\config_local.xml"
tree = et.ElementTree(file=configFilePath)
root = tree.getroot()

#IE 11 64bit Add key:HKEY_LOCAL_MACHINE\SOFTWARE\Wow6432Node\Microsoft\Internet Explorer\Main\FeatureControl\FEATURE_BFCACHE    name=iexplore.exe  value=DWORD(0)
#for detail:https://github.com/SeleniumHQ/selenium/wiki/InternetExplorerDriver#required-configuration
ieDriverPath = os.path.join(base_path,"bin","IEDriverServer.exe")
browser = webdriver.Ie(ieDriverPath)
browser.get("http://www.google.co.jp")
textbox = browser.find_element_by_name('q')
textbox.send_keys('python')
textbox.submit()
