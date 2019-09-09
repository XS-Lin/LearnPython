# 自動イベント進行
import numpy as np
import os
import cv2
import pyautogui
import sys
import time
import datetime
import PIL

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

def prepare(img_dir):
    if(os.path.isdir(img_dir)):
        files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) and f[-4:].upper() == '.JPG']
        for file in files:
            if len(data) == 0 or not any(x for x in data if x['name'] == file):
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

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = PIL.Image.fromarray(new_image)
    return new_image

def img_contains_cv(item):
    curr_img = pyautogui.screenshot()
    img2 = pil2cv(curr_img)
    kp_2, des_2 = akaze.detectAndCompute(img2, None)
    
    matches = bf.knnMatch(item['des_img'], des_2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    
    #cv2.namedWindow("Result", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
    #result_img = cv2.drawMatchesKnn(item['data'], item['kp_img'], img2, kp_2, good, None, flags=0)
    #cv2.imshow("Result", result_img)
    #cv2.waitKey(0)

    return len(good) / len(item['kp_img']) > 0.25

def do_lession():
    return
def do_work():
    return
def do_fes_daily():
    return
def do_fes_pre():
    menu = next(x for x in data if x['name'] == 'menu.JPG')
    if not (menu and img_contains_cv(menu)):
        #cv2.imshow('test',menu['data'])
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #pass
        return
    
    location = pyautogui.locateOnScreen(r'D:\Site\MyScript\python_test\scripts4game\doavv\input\fes_level1_pre.JPG',confidence=0.550)
    if location is None:
        return
    pyautogui.moveTo(location.left + 400,location.top, 2, pyautogui.easeInQuad)
    pyautogui.click()
    time.sleep(3)

    #fes_start = next(x for x in data if x['name'] == 'fes_start.JPG')
    #if not (fes_start and img_contains_cv(fes_start)):
    #    return
    location1 = pyautogui.locateOnScreen(r'D:\Site\MyScript\python_test\scripts4game\doavv\input\fes_start.JPG',confidence=0.550)
    if location1 is None:
        return
    pyautogui.moveTo(location1.left + location1.width//2 + 100 ,location1.top + location1.height // 2, 2, pyautogui.easeInQuad)
    pyautogui.click()
    time.sleep(20)
    
    location2 = pyautogui.locateOnScreen(r'D:\Site\MyScript\python_test\scripts4game\doavv\input\all_skip.JPG',confidence=0.550)
    if location2 is None:
        time.sleep(10)
        location2 = pyautogui.locateOnScreen(r'D:\Site\MyScript\python_test\scripts4game\doavv\input\all_skip.JPG',confidence=0.550)
        if location2 is None:
            return
    pyautogui.moveTo(location2.left + location2.width//2 ,location2.top + location2.height // 2, 2, pyautogui.easeInQuad)
    pyautogui.click()
    time.sleep(1)
    pyautogui.click()
    time.sleep(20)


    return

def do_fes_new():
    return

def main_process():
    #time.sleep(30)
    prepare(r'D:\Site\MyScript\python_test\scripts4game\doavv\input')
    try:
        while True:
            #save_path = os.path.join(input_dir,'file' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.jpg')
            #curr_img = pyautogui.screenshot(save_path)
            time.sleep(3)
         
            do_fes_pre()
            #kp_curr_img, des_curr_img = akaze.detectAndCompute(curr_img, None)
            # matches = bf.knnMatch(des_temp, des_samp, k=2)

            #cv2.imshow('test',pil2cv(curr_img))
            #cv2.waitKey(0)
            
            
    except KeyboardInterrupt:
        print('Stopped by user command.')
        return

main_process()

def test():
    img1 = cv2.imread(r'D:\Site\MyScript\python_test\scripts4game\doavv\input\fes_level1_pre.JPG',1)
    kp_1, des_1 = akaze.detectAndCompute(img1, None)
    img2 = cv2.imread(r'D:\Site\MyScript\python_test\scripts4game\doavv\image\file20190909202155.jpg',1)
    kp_2, des_2 = akaze.detectAndCompute(img2, None)
    matches = bf.knnMatch(des_1, des_2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.85*n.distance:
            good.append([m])

    cv2.namedWindow("Result", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
    #result_img = cv2.drawMatchesKnn(img1, kp_1, img2, kp_2, good, None, flags=0)
    result_img = cv2.drawKeypoints(img1,kp_1,None,flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Result", result_img)
    
    cv2.waitKey(0)
    return
#test()

def test1():
    prepare(r'D:\Site\MyScript\python_test\scripts4game\doavv\input')
    menu = next(x for x in data if x['name'] == 'menu.JPG')
    img2 = cv2.imread(r'D:\Site\MyScript\python_test\scripts4game\doavv\image\file20190909211806.jpg',1)
    kp_2, des_2 = akaze.detectAndCompute(img2, None)
    matches = bf.knnMatch(menu['des_img'], des_2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    result_img = cv2.drawMatchesKnn(menu['data'],menu['kp_img'], img2, kp_2, good, None, flags=0)
    #result_img = cv2.drawKeypoints(menu['data'],menu['kp_img'],None,flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Result", result_img)
    cv2.waitKey(0)
    print(len(good))
    print(len(menu['kp_img']))
    print(len(good) / len(menu['kp_img']) > 0.25)
    return
#test1()

def test2():
    time.sleep(10)
    location = pyautogui.locateOnScreen(r'D:\Site\MyScript\python_test\scripts4game\doavv\input\fes_level1_pre.JPG',confidence=0.600)
    if location:
        print(location)
    return
#test2()