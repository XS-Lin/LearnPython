import os
from io import BytesIO
import xml.etree.ElementTree as et

from selenium import webdriver
from PIL import Image as pillowImage
import openpyxl
from openpyxl.drawing.image import Image as pyxlImage

base_path = r"C:\Users\linxu_000\Desktop\python script"
ieDriverPath = os.path.join(base_path,"bin","IEDriverServer.exe")
browser = webdriver.Ie(ieDriverPath)

browser.get("http://www.google.co.jp")
textbox = browser.find_element_by_name('q')
textbox.send_keys('python')
textbox.submit()

screenShot = browser.get_screenshot_as_png()
im = pillowImage.open(BytesIO(screenShot))
#im.save('path')

img = pyxlImage(im)

wb = openpyxl.load_workbook(filename=r'C:\Users\linxu_000\Desktop\python script\testFolder\test1.xlsx')
ws = wb["Sheet1"]
ws.add_image(img,'D3')
wb.save(r"C:\Users\linxu_000\Desktop\python script\testFolder\test3.xlsx")
