# cmdline parameter: "account","password"
import sys
import os
from selenium import webdriver

# BROWSER_INFO: Chrome 76.0.3809.87
# DRIVER_INFO : ChromeDriver 76.0.3809.68 [DOWNLOAD](https://chromedriver.storage.googleapis.com/index.html?path=76.0.3809.68/)
driver = webdriver.Chrome(os.path.join('..','bin','chromedriver_win32','chromedriver.exe'))
if len(sys.argv) <= 1:
    driver.get("mail.yahoo.co.jp")
else:
    arg = sys.argv[1]
    if arg.endswith('@126.com'):
        driver.get("mail.126.com")


