# 自動イベント進行
import numpy as np
import os
import cv2
import pyautogui
import time
import datetime
import PIL
import json
from matplotlib import pyplot as plt

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
temp_dir = r'D:\Test'
button_pos = {
    'btn_menu_main_Fes':(83,286),
    'btn_menu_main_Girls':(83,432),

    'btn_menu_lv1_Suggestions':(1377,216),
    'btn_menu_lv1_Suggestions_new':(1622,649),
    'btn_menu_lv1_Suggestions_pre':(1622,754),
    'btn_menu_lv1_Suggestions_pre_1':(1622,649),

    'btn_fes_Start':(1480,1000),
    'btn_fes_SkipAll':(1500,1000),

    'btn_confirm':(1075,870),

    'add_active_point':(882,700),
}
def prepare(img_dir):
    if(os.path.isdir(img_dir)):
        files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) and f[-4:].upper() == '.JPG']
        for file in files:
            if len(data) == 0 or not any(x for x in data if x['name'] == file):
                img = cv2.imread(os.path.join(img_dir, file), 1)
                kp_img, des_img = akaze.detectAndCompute(img, None)
                data.append({'name':file,'data':img,'kp_img':kp_img,'des_img':des_img,'path':os.path.join(img_dir, file)})
    return

def pil2cv(image):
    new_image = np.array(image,dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(image):
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = PIL.Image.fromarray(new_image)
    return new_image

def save_data():
    if len(data) > 0:
        with open(os.path.join(temp_dir,'data.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def load_data():
    if os.path.isfile(os.path.join(temp_dir,'data.json')):
        with open(os.path.join(temp_dir,'data.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
    if data is None:
        data = []

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

def wait_until(condition,params=None,timeout=60):
    t = 0
    if callable(condition):
        while t < timeout:
            if condition(params) == True:
                break
            time.sleep(1)
            t = t + 1
    return t < timeout

def add_active_point(flag=True):
    if flag:
        active_point_confirm = next(x for x in data if x['name'] == 'active_point_confirm.JPG')
        if active_point_confirm and img_contains_cv(active_point_confirm):
            x,y = button_pos['add_active_point']
            pyautogui.moveTo(x, y, 2, pyautogui.easeInQuad)
            pyautogui.click()
            time.sleep(1)
            x,y = button_pos['btn_confirm']
            pyautogui.moveTo(x, y, 2, pyautogui.easeInQuad)
            pyautogui.click()
            time.sleep(1)
    return

def do_lession():
    return
def do_work():
    return
def do_fes_daily():
    return
def do_fes_pre():
#    menu = next(x for x in data if x['name'] == 'menu.JPG')
#    if menu and img_contains_cv(menu):       
#        x,y = button_pos['btn_menu_main_Fes']
#        pyautogui.moveTo(x, y, 2, pyautogui.easeInQuad)
#        pyautogui.click()
#        time.sleep(3)

    fes_level1_pre = next(x for x in data if x['name'] == 'fes_level1_pre.JPG')
    # "akaze.detectAndCompute" dose not work
    location = pyautogui.locateOnScreen(fes_level1_pre['path'],confidence=0.550) 
    if location:
        x,y = button_pos['btn_menu_lv1_Suggestions_pre']
        pyautogui.moveTo(x, y, 2, pyautogui.easeInQuad)
        pyautogui.click()
        time.sleep(1)

    fes_start = next(x for x in data if x['name'] == 'fes_start.JPG')
    if fes_start and img_contains_cv(fes_start):
        x,y = button_pos['btn_fes_Start']      
        pyautogui.moveTo(x, y, 2, pyautogui.easeInQuad)
        pyautogui.click()
        time.sleep(15)
        pyautogui.click()
        time.sleep(3)
        pyautogui.click()
        time.sleep(1)
        pyautogui.click()
        time.sleep(1)

    btn_fes_SkipAll = next(x for x in data if x['name'] == 'all_skip.JPG')
    if btn_fes_SkipAll and img_contains_cv(btn_fes_SkipAll):
        x,y = button_pos['btn_fes_SkipAll']      
        pyautogui.moveTo(x, y, 1, pyautogui.easeInQuad)
        pyautogui.click()
        time.sleep(1)

    btn_confirm = next(x for x in data if x['name'] == 'btn_confirm.JPG')
    if btn_confirm and img_contains_cv(btn_confirm):
        x,y = button_pos['btn_confirm']      
        pyautogui.moveTo(x, y, 1, pyautogui.easeInQuad)
        pyautogui.click()
        time.sleep(5)
        pyautogui.click()
        time.sleep(3)
        pyautogui.click()
        time.sleep(2)
        pyautogui.click()
        time.sleep(2)
        pyautogui.click()
        time.sleep(2)
        pyautogui.click()
        time.sleep(2)
        pyautogui.click()
        time.sleep(2)
        pyautogui.moveTo(x+100, y+100, 1, pyautogui.easeInQuad)
        pyautogui.click()
        time.sleep(2)
        pyautogui.click()
        time.sleep(10)

    return

def do_fes_new():
    return

def main_process():
    time.sleep(5)
    prepare(r'D:\Site\MyScript\python_test\scripts4game\doavv\input')
    try:
        while True:
            time.sleep(3)
            #add_active_point()
            do_fes_pre()
    except KeyboardInterrupt:
        print('Stopped by user command.')
        return

#main_process()

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
    menu = next(x for x in data if x['name'] == 'btn_confirm.JPG')
    img2 = cv2.imread(r'D:\Site\MyScript\python_test\scripts4game\doavv\image\file20190909202116.jpg',1)
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

def test3():
    img_dir = r'D:\Site\MyScript\python_test\scripts4game\doavv\image\file20190910210035.jpg'
    img = cv2.imread(img_dir,1)
    img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img1)
    plt.show()
    return
#test3()

def test4():
    time.sleep(5)
    try:
        while True:
            save_path = os.path.join(r'D:\Site\MyScript\python_test\scripts4game\doavv\image','file' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.jpg')
            pyautogui.screenshot(save_path)
            time.sleep(3)
    except KeyboardInterrupt:
        print('Stopped by user command.')
    return
#test4()