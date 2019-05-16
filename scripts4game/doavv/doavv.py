# 自動イベント進行
import pyautogui
import sys
import time

# ボタン位置
# 
#
#
#
#
#
#
#
def autoEvent():
    time.sleep(1)
    pyautogui.moveTo(826, 778, 1, pyautogui.easeInBounce)
    pyautogui.moveTo(826, 778, 1, pyautogui.easeInElastic)
    pyautogui.click()
    return

print('Press Ctrl-C to quit.')
pyautogui.PAUSE = 0.2
try:
    time.sleep(10) # 起動後の10秒は何もしない
    while True:
        autoEvent()
except KeyboardInterrupt:
    print('\n')