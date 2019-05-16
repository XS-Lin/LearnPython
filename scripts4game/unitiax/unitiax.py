# 自動「再挑戦」ボタンを押す
import pyautogui
import sys
import time

# 826,778 再挑戦
# 1378,777 NEXT
# 1111,665 使用
# 832,704 はい

print('Press Ctrl-C to quit.')
time.sleep(10)

def autoUpgrade():
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(1378, 777, 2, pyautogui.easeInBounce)
    pyautogui.click()
    time.sleep(8)
    pyautogui.click()
    time.sleep(8)
    pyautogui.moveTo(826, 778, 2, pyautogui.easeInBounce)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(1111, 665, 2, pyautogui.easeInBounce)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(832, 704, 2, pyautogui.easeInBounce)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(826, 778, 2, pyautogui.easeInBounce)
    time.sleep(1)
    pyautogui.click()
    # SPEED X2 用
    # time.sleep(90) 
    # SPEED X3 用
    time.sleep(70) 

try:
    while True:
        
        time.sleep(1)
        
        x, y = pyautogui.position()
        positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
        print(positionStr, end='')
        print('\b' * len(positionStr), end='', flush=True)

        autoUpgrade()
        
except KeyboardInterrupt:
    print('\n')


