import openpyxl
from copy import copy

wb = openpyxl.load_workbook(r"C:\Users\linxu_000\Desktop\python script\testFolder\test1.xlsx")

wb["Sheet2"]["B4"].value = wb["Sheet1"]["B2"].value
wb["Sheet2"]["B4"].font = copy(wb["Sheet1"]["B2"].font)
wb["Sheet2"]["B4"].border = copy(wb["Sheet1"]["B2"].border)
wb["Sheet2"]["B4"].fill = copy(wb["Sheet1"]["B2"].fill)
wb["Sheet2"]["B4"].number_format = copy(wb["Sheet1"]["B2"].number_format)
wb["Sheet2"]["B4"].protection = copy(wb["Sheet1"]["B2"].protection)
wb["Sheet2"]["B4"].alignment = copy(wb["Sheet1"]["B2"].alignment)

wb.save(r"C:\Users\linxu_000\Desktop\python script\testFolder\test1copy.xlsx")