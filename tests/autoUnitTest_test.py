import os
from io import BytesIO
import xml.etree.ElementTree as et

from selenium import webdriver
from selenium.webdriver.common.alert import Alert
from PIL import Image as pillowImage
import openpyxl
from openpyxl.drawing.image import Image as pyxlImage

#----------------------------Setting----------------------------
base_addr = "http://127.0.0.1:8080/"
target = [
    "sqlijc",
    "ureupload",
    "xss"
]

ie_driver_path = r"C:\Users\linxu_000\Desktop\python script\bin\IEDriverServer.exe"
chrome_driver_path = r"C:\Users\linxu_000\Desktop\python script\bin\chromedriver_win32\chromedriver.exe"

excel_template_path = r"C:\Users\linxu_000\Desktop\python script\testFolder\test1.xlsx"
excel_template_start_cell = "D3"
excel_template_row_increment = 40
excel_template_sheet_name = "Sheet1"

excel_output_pic_cache_size = 10
excel_output_path = r"C:\Users\linxu_000\Desktop\python script\testFolder\test3.xlsx"

upload_test_file = r"C:\Users\linxu_000\Desktop\python script\testFolder\testPic_klaudia.jpg"
#----------------------------Process--------------------
driver = webdriver.Ie(chrome_driver_path)
wb = openpyxl.load_workbook(filename=excel_template_path)
ws = wb[excel_template_sheet_name]

driver.get(base_addr + target[0])
driver.find_element_by_name("name").send_keys("Mark")
driver.find_element_by_name("password").send_keys("password")

screenShot = driver.get_screenshot_as_png()
img = pillowImage.open(BytesIO(screenShot))
#img.save('path')
xImg = pyxlImage(img)
ws.add_image(xImg,"C1")

driver.find_element_by_xpath(r"./html/body/form/input[@type='submit']").click()

screenShot = driver.get_screenshot_as_png()
img = pillowImage.open(BytesIO(screenShot))
xImg = pyxlImage(img)
ws.add_image(xImg,"C51")

driver.get(base_addr + target[1])
driver.find_element_by_name("file").send_keys(upload_test_file)
driver.find_element_by_xpath(r"//input[@type='submit']").click()

screenShot = driver.get_screenshot_as_png()
img = pillowImage.open(BytesIO(screenShot))
xImg = pyxlImage(img)
ws.add_image(xImg,"C101")

driver.get(base_addr + target[2])
driver.find_element_by_name("string").send_keys("abc")
driver.find_element_by_xpath(r"//input[@type='submit']").submit()
driver.find_element_by_name("string").send_keys(r">tpircs/<;)eikooc.tnemucod(trela>tpIrcs<")
driver.find_element_by_xpath(r"//input[@type='submit']").click()
Alert(driver).accept()

wb.save(excel_output_path)
wb.close()
