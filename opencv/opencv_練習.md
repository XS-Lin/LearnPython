# OpenCV練習 #

## Download&Install ##

* Windows10(C++/VS2019)
  * opencv-4.1.1-vc14_vc15.exe
  * 展開先 D:\opencv4_1_1\
  * 環境変数追加

    ~~~powershell
    [System.Environment]::SetEnvironmentVariable("PATH", "D:\opencv4_1_1\opencv\;" + [Environment]::GetEnvironmentVariable('PATH', 'User'), "User")
    ~~~

  * インストール確認

    参考記事[OpenCV 4.1.1をVisual Studio 2019から使用する時の手順](https://qiita.com/h-adachi/items/aad3401b8900438b2acd)

      1. プロジェクト作成
         * 言語:C++
         * プラットフォーム:Windows
         * タイプ:コンソール アプリ

      1. プロジェクト・プロパティ設定
         * 構成:Debug
         * プラットフォーム:x64
         * 構成プロパティ/デバッグ/環境:PATH=D:\opencv4_1_1\build\x64\vc15\bin;%PATH%
         * 構成プロパティ/VC++ ディレクトリ/インクルード ディレクトリ:D:\opencv4_1_1\build\include
         * 構成プロパティ/VC++ ディレクトリ/ライブラリ ディレクトリ:D:\opencv4_1_1\build\x64\vc15\lib
         * 構成プロパティ/リンカー/入力/追加の依存ファイル:opencv_world411d.lib

      1. プログラム

         ~~~cpp
         #include <opencv2/opencv.hpp>
         using namespace cv;
         int main()
         {
             Mat image = Mat::zeros(100, 100, CV_8UC3);
             imshow("", image);
             waitKey(0);
         }
         ~~~

* Windows10(Python3.7)

  * pip install opencv-python

    ~~~python
    import cv2

    src = cv2.imread('sample.png')
    cv2.imshow('test', src)
    cv2.waitKey(0)
    ~~~

* CentOs7.5 (Python3.7.4)

  * python3.7.4 インストール

    TODO:

  * pip3 install opencv-python
  * python3 test_cv.py

    ~~~python
    #!/bin/usr/python3
    import cv2

    src = cv2.imread('sample.png')
    cv2.imshow('test', src)
    cv2.waitKey(0)
    ~~~

## Tips ##

VS Code使用の時に、エラー「**pylint(no-member):Module 'cv2' has no 'imread' member**」が発生しました。
設定ファイルに以下を追加して、解決できた。

~~~json
"python.linting.pylintArgs": ["--generate-members"]
~~~

## チュートリアル ##

参考[OpenCV-Pythonチュートリアル](http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_tutorials.html)

TODO:以下を理解してみる
[物体追跡](http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#id3)

[GetDesktopAsVideo](https://stackoverflow.com/questions/34466993/opencv-desktop-capture)
[Capture video data from screen in Python](https://stackoverflow.com/questions/35097837/capture-video-data-from-screen-in-python/43560140)
[Pythonで簡易デスクトップキャプチャ](https://scienceboy.jp/88io/2018/12/python-quick-capture/)
[Python/OpenCVでデスクトップキャプチャ - 画像処理速報](http://opencv.blog.jp/python/desktop_capture)

[TensorFlow入門](https://www.atmarkit.co.jp/ait/subtop/features/di/introtensorflow_index.html)

## 練習1 ##

1. ゲーム「DOA vv」の自動プレーツール

   1. 機能

      1. Windowモードのゲームの画像を取得する。
      1. ゲームの画像をもとに、次の選択肢を決める。


