import os
from selenium import webdriver
import xml.etree.ElementTree as et
base_path = ""
configFilePath = "config.xml"
#base_path=r"C:\Users\linxu_000\Desktop\python script"
#configFilePath=r"C:\Users\linxu_000\Desktop\python script\config.xml"
tree = et.ElementTree(file=configFilePath)
root = tree.getroot()

ieDriverPath = os.path.join(base_path,"bin","IEDriverServer.exe")
browser = webdriver.Ie(ieDriverPath)

