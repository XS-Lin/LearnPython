# 自動イベント進行
import numpy as np
import os
import cv2
import pyautogui
import sys
import time
import datetime

# 準備
# 1.検知対象画像準備
# 2.行動パターン定義
# メイン
# 1.デスクトップキャプチャーをとる(pyautogui 約0.1s/1枚))
# 2.Opencv形式へ変換
# 3.特徴比較
#    参考:https://m12watanabe1a.hatenablog.com/entry/2018/10/14/201503
#         http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_feature2d/py_matcher/py_matcher.html#matcher
# 4.行動を行う

data = []
akaze = cv2.AKAZE_create()
bf = cv2.BFMatcher()
input_dir = r'D:\Site\MyScript\python_test\scripts4game\doavv\image'

def prepare(img_dir):
    if(os.path.isdir(img_dir)):
        file = [f for f in os.path.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) and f[-4:].upper() == '.PNG']
        if len(data) > 0 and not any(x for x in data if x['name'] == file):
            img = cv2.imread(os.path.join(img_dir, file), 1)
            kp_img, des_img = akaze.detectAndCompute(img, None)
            data.append({'name':file,'data':img,'kp_img':kp_img,'des_img':des_img})
    return

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image,dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def status_machine(s=0,r=0):
    startState = 0
    currState = s
    endState = 9
    if (currState > endState or currState <startState ):
        return
    

    return

def main_process():
    try:
        while True:
            #save_path = os.path.join(input_dir,'file' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.jpg')
            #curr_img = pyautogui.screenshot(save_path)
            time.sleep(3)

            curr_img = pyautogui.screenshot()
            #kp_curr_img, des_curr_img = akaze.detectAndCompute(curr_img, None)
            # matches = bf.knnMatch(des_temp, des_samp, k=2)

            #cv2.imshow('test',pil2cv(curr_img))
            #cv2.waitKey(0)
            
    except KeyboardInterrupt:
        print('Stopped by user command.')
        return

main_process()


# ボタン位置
# 
#
#
#
#
#
#
#
#def autoEvent():
#    time.sleep(1)
#    pyautogui.moveTo(826, 778, 1, pyautogui.easeInBounce)
#    pyautogui.moveTo(826, 778, 1, pyautogui.easeInElastic)
#    pyautogui.click()
#    return
#
#print('Press Ctrl-C to quit.')
#pyautogui.PAUSE = 0.2
#try:
#    time.sleep(10) # 起動後の10秒は何もしない
#    while True:
#        autoEvent()
#except KeyboardInterrupt:
#    print('\n')