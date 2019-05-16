import keyboard
import sys
import time
import random

# 「x」キーを押しているときに、連続「x」を押すように変更
def keepKeyToPressKey(keyCode):
    print('Press Ctrl-C to quit.')
    try:
        while True:
            randomTime = 1 # キー押下していない場合、1s待ち
            if keyboard.is_pressed('x'):
                randomTime = random.randrange(3,10) / 100 # キー押下間隔 0.04-0.10s (1秒あたり10~25回押す)
                keyboard.send('y')
            time.sleep(randomTime)    
    except KeyboardInterrupt:
        print('\n')
        return
keepKeyToPressKey(b'x')
