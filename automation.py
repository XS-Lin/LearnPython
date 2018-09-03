""" 
"""
import time
import os
import subprocess
import io
import sys
from pprint import pprint
from zapv2 import ZAPv2
from selenium import webdriver
import xml.etree.ElementTree as et
from PIL import Image as pillowImage
#----------------------------Target Site---------------------------------------
base_target = "http://localhost:8080/"
target_urls = [
"xss",
"sqlijc",
"nullbyteijct",
"xxe",
"ursupload",
"login"
]
#----------------------------Settings------------------------------------------
api_key = "doofog4j8dkamn23ocqpdhjm34"
zap_proxy = "http://127.0.0.1:8180"

ie_driver_path = os.path.join("bin","IEDriverServer.exe")
report_folder =  "testFolder"

#----------------------------Init----------------------------------------------
driver = webdriver.Ie(ie_driver_path)
driver.get(base_target)
zap = ZAPv2(apikey=api_key, proxies={'http': zap_proxy, 'https': zap_proxy})
time.sleep(3)

#----------------------------Process-------------------------------------------

target_page = base_target + target_urls[1]
driver.get(target_page)

scanid = zap.ascan.scan(target_page)
while (int(zap.ascan.status(scanid)) < 100):
    print('Scan progress %: ' + zap.ascan.status(scanid))
    time.sleep(5)
print('Scan completed:' + target_page)

report_file_path=  os.path.join(report_folder, target_urls[1] + ".html")
report_file = io.open(report_file_path, 'w+', encoding='utf8')
report_file.write(zap.core.htmlreport())
report_file.close()
print('Report:' + report_file_path)

#----------------------------Clear---------------------------------------------
zap.core.shutdown()
