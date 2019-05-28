import pyautogui
import keyboard
import sys
import time

pyautogui.PAUSE = 0.05
print('Press Ctrl-C to quit.')

def recordMousePositionWhenPressP():
    try:
        while True:
            keyboard.wait('p')
            x, y = pyautogui.position()
            positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
            print(positionStr, end='\n')
            #print('\b' * len(positionStr), end='', flush=True)    
    except KeyboardInterrupt:
        print('\n')
recordMousePositionWhenPressP()