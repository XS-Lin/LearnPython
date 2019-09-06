import cv2
import numpy as np
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