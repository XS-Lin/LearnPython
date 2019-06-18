# 機能:
#   ネットワーク通信により、
#     Excel読み取りのリクエスト受信し、Excelの内容を戻します。
#     Excel書き込みのリクエスト受信し、Excelの内容を出力します。
#   なお、Officeのインストールする必要はありません。
# 使用例:
#   WebApplication等と組み合わせ、ServerサイドでExcelを処理します。
#   起動コマンド(python3はPython3.7のシンボリックリンク)
#     python3 excel_tool_by_openpyxl.py
#   テストマンド(正常の場合は「Processing」表示されます)
#     例:base
#       curl http://ip:PORT/test
#     例:ブラウザ
#       http://ip:PORT/testを開く
#   処理コマンド(HTTPリクエスト送信)
#     例:bash
#       curl -H 'Content-Type:application/json; charset=utf-8' -d "{"key":"70949d2b88c04e869b074597700dd22d","fileName":"test.xml","process_method":"fun1"}" http://ip:PORT/read
#     例:bash
#       curl -H 'Content-Type:application/json' -d "{"key":"70949d2b88c04e869b074597700dd22d","fileName":"test.xml","process_method":"fun2"}" http://ip:PORT/write
# 検証環境
#   CentOs7.5
#   Python3.7.3
#   Windows10
#   Excel2019
# 注意:
#   1.大きいサイズのExcel(例えば100M以上)は想定していません。
#   2.入力ExcelはExcel2010以上が必要になります。(xlsx)
#   3.出力のExcelはExcel2010以上で表示できます。
#   4.使用範囲は内部ネットワークと想定しています。
#   5.デフォルト処理以外の場合は拡張が必要になります。
import os
import socket
import csv
import openpyxl
from copy import copy
# -----------------------------------------------------------------------------
# アクセスキー(ヘッダにキーが未設定のtest以外のリクエストは無視する)
ACCESS_KEY = "70949d2b88c04e869b074597700dd22d"
# コマンド受信ポート
PORT = 10001
# ファイル読み取り時のヘーズフォルダ(外層はアクセスしない)
BASE_FOLDER = ""
# 中間ファイル保存用フォルダ
WORK_FOLDER = "/"
# 入力ファイルのフォルダ(WebAppのアップロードフォルダ)
INPUT_FOLDER = "~/UPLOAD"
# 出力ファイルのフォルダ(WebAppのダウンロード用フォルダ)
OUTPUT_FOLDER = "~/DOWNLOAD"
# -----------------------------------------------------------------------------

wb = openpyxl.load_workbook(r"C:\Users\linxu_000\Desktop\python script\testFolder\test1.xlsx")

wb["Sheet2"]["B4"].value = wb["Sheet1"]["B2"].value
wb["Sheet2"]["B4"].font = copy(wb["Sheet1"]["B2"].font)
wb["Sheet2"]["B4"].border = copy(wb["Sheet1"]["B2"].border)
wb["Sheet2"]["B4"].fill = copy(wb["Sheet1"]["B2"].fill)
wb["Sheet2"]["B4"].number_format = copy(wb["Sheet1"]["B2"].number_format)
wb["Sheet2"]["B4"].protection = copy(wb["Sheet1"]["B2"].protection)
wb["Sheet2"]["B4"].alignment = copy(wb["Sheet1"]["B2"].alignment)

wb.save(r"C:\Users\linxu_000\Desktop\python script\testFolder\test1copy.xlsx")

#TODO
#read,write excel with style