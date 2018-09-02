import time
import os
import subprocess
import io
from pprint import pprint
from zapv2 import ZAPv2
from selenium import webdriver
import xml.etree.ElementTree as et
from PIL import Image as pillowImage

base_target = "http://localhost:8080"
target_urls = [
"/xss",
"/sqlijc",
"/nullbyteijct",
"/xxe",
"/ursupload",
"/login"
]

ieDriverPath = os.path.join("bin","IEDriverServer.exe")
driver = webdriver.Ie(ieDriverPath)
driver.get(base_target)
time.sleep(1)

zap = ZAPv2(apikey='doofog4j8dkamn23ocqpdhjm34', proxies={'http': 'http://127.0.0.1:8180', 'https': 'http://127.0.0.1:8180'})

target_page = base_target + target_urls[1]
driver.get(target_page)

scanid = zap.ascan.scan(target_page)
while (int(zap.ascan.status(scanid)) < 100):
    print('Scan progress %: ' + zap.ascan.status(scanid))
    time.sleep(5)

print('Scan completed')

temp_report_folder =  os.path.join("testFolder","zapreport.html")
fHTML = io.open(temp_report_folder,'w+',encoding='utf8')
fHTML.write(zap.core.htmlreport())
fHTML.close()