# 

class TestContext:
    def __init__(self,connString = '', logFilePath = '', evidenceExcelFile = ''):
        self.useOracle = False
        self.oraConnection = None
        self.oraCursor = None
        self.useLog = False
        self.logMessage = []
        self.logFile = None
        self.useExcel = False
        self.excelFile = None
        
        if len(connString) > 0:
            return

        if len(logFilePath) > 0:
            return

        if len(evidenceExcelFile) > 0:
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



    