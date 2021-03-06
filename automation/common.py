import io
import sys
import os
import json
import openpyxl
import cx_Oracle
import time
from selenium import webdriver
from PIL import Image as pillowImage

class Context:
    def __init__(self,connString = '', logFilePath = '', evidenceExcelFile = ''):
        self.useOracle = False
        self.oraConnection = None
        self.oraCursor = None
        self.useLog = False
        self.logMessage = []
        self.logFile = None
        self.useExcel = False
        self.execlPosition = 'A1'
        self.excelFile = None
        self.driver = None

        if len(connString) > 0:
            try:
                self.useOracle = True
            except:
                self.useOracle = False
            return

        if len(logFilePath) > 0:
            try:
                self.useLog = True
            except:
                self.useLog = False
            return
        else:
            return

        if len(evidenceExcelFile) > 0:
            try:
                self.useExcel = True
            except:
                self.useExcel = False
            return

    def addLogMessage(self,message = ''):
        return
    
    def flushLog(self):
        return

    def execute(self,sql,named_params):
        return

    def begin(self):
        return
    
    def commit(self):
        return
    
    def rollback(self):
        return

    def getCapture(self):
        return 

    def saveCapture(self):
        return

    def checkInputData(self):
        return

def bindDataToPage(context,pageId):
    return True