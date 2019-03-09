import numpy as np
import xlwings as xw

def output_hello():
    wb = xw.Workbook.caller()
    n = xw.Range('Sheet1', 'B1').value
    xw.Range('Sheet1', 'C3').value = n + ', hello!'