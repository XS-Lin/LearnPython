import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

def base():
    imgPath = r"C:\Users\linxu\Pictures\Desktop\icon_sen4_03.jpg"
    img1 = cv2.imread(imgPath,1) # cv2.IMREAD_COLOR
    img0 = cv2.imread(imgPath,0) # cv2.IMREAD_GRAYSCALE
    imga = cv2.imread(imgPath,-1) # cv2.IMREAD_UNCHANGED
    
    #plt.imshow(img0, cmap = 'gray', interpolation = 'bicubic')
    #plt.xticks([]), plt.yticks([])
    #plt.show()
    
    print(img1.shape) # (400,400,3) # 列の数，行の数，チャンネル数(色相数)
    print(img1.size) # 480000 # 画素数
    print(img1.dtype) # uint8
    
    img1[100,100] # [174 171 255] # [blue,green,red]
    
    img1[100,100,0] # BLUE画素値
    img1.item(100,100,0) # BLUE画素値
    
    # 画像一部取得
    x = img1[100:150,200:270] # Numpy Index: X[start:end:step,start:end:step]
    img1[0:50,0:70] = x
    
    # 画像の色成分の分割と統合
    b,g,r = cv2.split(img1)
    img1 = cv2.merge((b,g,r))
    
    b = img1[:,:,0]
    img1[:,:,2] = 0
    
    # 画像の境界領域を作る
    #replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE) # 最後の要素が繰り返し  aaaaaa|abcdefgh|hhhhhhh
    #reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT) # 鏡に写したかのように境界 fedcba|abcdefgh|hgfedcb
    #reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101) # 鏡に写したかのように境界 gfedcb|abcdefgh|gfedcba
    #wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP) # cdefgh|abcdefgh|abcdefg
    #constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,0,0]) # 単一色の境界,次の引数で色の指定
    
    #plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
    #plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
    #plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
    #plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
    #plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
    #plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
    
    #plt.show()

def compute():
    cv2.setUseOptimized(False) # Default:true 

    img1 = cv2.imread(r"C:\Users\linxu\Pictures\Desktop\icon_sen4_03.jpg",1)
    img2 = cv2.imread(r"C:\Users\linxu\Pictures\Desktop\icon_sen4_01.jpg",1)
    
    e1 = cv2.getTickCount()

    dst = cv2.addWeighted(img1,0.7,img2,0.3,0) # dst = \alpha \cdot img1 + \beta \cdot img2 + \gamma

    e2 = cv2.getTickCount()
    time = (e2 - e1)/ cv2.getTickFrequency()

    cv2.imshow('dst',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(time)
    return
#compute()
# Pythonのスカラー計算はNumpyのスカラー計算より高速
# 一般的にOpenCVの関数はNumpyの関数より高速,例外：コピーではなくデータの値を見る時はNumpyの方が高速

def ChangeColorSpace():
    # BGR(blue,green,red) <--> HSV(Hue,Saturation,Value) # Hue->色相，Saturation(Chroma)->彩度，Value(Lightness)->明度
    cap = cv2.VideoCapture(0)
    
    while(1):
    
        # Take each frame
        _, frame = cap.read()
    
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        # define range of blue color in HSV
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
    
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)
    
        cv2.imshow('frame',frame)
        cv2.imshow('mask',mask)
        cv2.imshow('res',res)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    
    cv2.destroyAllWindows()
    # BGR <--> Gray

    return
ChangeColorSpace()

def test_feature_detection():
    template_path = "D:\\Test\\"
    template_filename = "you_shitsu.jpg"

    sample_path = "D:\\Test\\"
    sample_filename = "floor_plan_001.jpg"

    result_path = "D:\\Test\\"
    result_name = "perfect.jpg"

    akaze = cv2.AKAZE_create()

    # 文字画像を読み込んで特徴量計算
    expand_template=2
    whitespace = 20
    template_temp = cv2.imread(template_path + template_filename, 0)
    height, width = template_temp.shape[:2]
    template_img=np.ones((height+whitespace*2, width+whitespace*2),np.uint8)*255
    template_img[whitespace:whitespace + height, whitespace:whitespace+width] = template_temp
    template_img = cv2.resize(template_img, None, fx = expand_template, fy = expand_template)
    kp_temp, des_temp = akaze.detectAndCompute(template_img, None)
    
    # 間取り図を読み込んで特徴量計算
    expand_sample = 2
    sample_img = cv2.imread(sample_path + sample_filename, 0)
    sample_img = cv2.resize(sample_img, None, fx = expand_sample, fy = expand_sample)
    kp_samp, des_samp = akaze.detectAndCompute(sample_img, None)
    
    # 特徴量マッチング実行
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_temp, des_samp, k=2)
    
    # マッチング精度が高いもののみ抽出
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)
    
    # マッチング結果を描画して保存
    cv2.namedWindow("Result", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
    #result_img = cv2.drawMatchesKnn(template_img, kp_temp, sample_img, kp_samp, good, None, flags=0)
    #cv2.imshow("Result", result_img)
    # cv2.imwrite(result_path + result_name, result_img)

    src_pts = np.float32([ kp_temp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp_samp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = template_img.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    sample_img = cv2.polylines(sample_img,[np.int32(dst)],True,0,5, cv2.LINE_AA)
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    result_img = cv2.drawMatches(template_img, kp_temp, sample_img, kp_samp, good,None, **draw_params)
    cv2.imshow("Result", result_img)

    #x = cv2.drawKeypoints(template_img,kp_temp,None,flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    #cv2.imshow("Result",x)
    cv2.waitKey(0) 

    return

#test_feature_detection()

def wait_until(condition,params=None,timeout=60):
    t = 0
    if callable(condition):
        while t < timeout:
            if condition(params) == True:
                break
            time.sleep(1)
            t = t + 1
    return t < timeout

def test(x):
    return x > 5

#wait_until(test,6,10)

def showImagePlot(img_path):
    img = cv2.imread(img_path,1)
    img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img1)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    return
showImagePlot(r'C:\Users\linxu\Desktop\project\python_test\scripts4game\doavv\image\file20190909202155.jpg')