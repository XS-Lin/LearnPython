# 自動「再挑戦」ボタンを押す
import pyautogui
import sys
import time

#print('Press Ctrl-C to quit.')

def autoUpgrade():
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(1378, 777, 2, pyautogui.easeInBounce)
    pyautogui.click()
    time.sleep(8)
    pyautogui.click()
    time.sleep(8)
    pyautogui.moveTo(826, 778, 2, pyautogui.easeInBounce)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(1111, 665, 2, pyautogui.easeInBounce)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(832, 704, 2, pyautogui.easeInBounce)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(826, 778, 2, pyautogui.easeInBounce)
    time.sleep(1)
    pyautogui.click()
    # SPEED X2 用
    # time.sleep(90) 
    # SPEED X3 用
    time.sleep(70) 

#try:
#    while True:
#        
#        time.sleep(1)
#        
#        x, y = pyautogui.position()
#        positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
#        print(positionStr, end='')
#        print('\b' * len(positionStr), end='', flush=True)
#
#        #autoUpgrade()
#
#except KeyboardInterrupt:
#    print('\n')

def getPosition():
    try:
        while True:
            time.sleep(1)
                
            x, y = pyautogui.position()
            positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
            print(positionStr, end='')
            print('\b' * len(positionStr), end='', flush=True)
        
            #autoUpgrade()
    except KeyboardInterrupt:
        print('\n')

    return


class AutoSet:
    def __init__(self):
        # メーンメニュー
        self.btn_Arena = (1366,347) # アリーナ
        self.btn_Character = (1155,679) # キャラ
        self.btn_Equip = (1297,677) # 装備品
        self.btn_GetItem = (1073,785) # ガチャ
        self.btn_Guild = (1267,785) # ギルド
        self.btn_Friend = (1367,789) # フレンド
        self.btn_Mission = (522,358) # ミッション
        self.btn_Presents = (478,517) # PRESENTS
        self.btn_LimitQuest = (1272,242) # リミテッドクエスト
        self.btn_Quest = (1387,579) # クエスト
        # アリーナ
        self.btn_ArenaReady = (1247,800) # アリーナ準備
        self.btn_Battle = (1340,780) # バトル
        self.btn_ArenaTop = (1367,773) # アリーナトップ
        self.btn_OK_2 = (954,707) # OK
        # クエスト
        self.btn_SpQuest = (1332,685) # SPクエスト
        self.btn_EquipmentQuest = (560,624) # 装備品クエスト
        self.btn_EqQuest_Weapon = (572,552) # 虹に立つ城 剣の間
        self.btn_Next = (1397,778) # 進行
        self.btn_Challenge = (826,778) # 再挑戦
        self.btn_QuestSelect = (1066,778) # クエスト選択へ
        # キャラ
        self.btn_Enforce = (592,387) # 強化
        # ガチャ
        # 装備品
        # ギルド
        self.btn_GuildQuest = (660,778) # ギルドクエスト
        self.btn_GuildQuestReady = (1278,792) # ギルドクエスト準備
        self.btn_Battle = (1325,777) # Battle
        self.btn_Top = (1378,777) # TOPへ
        # フレンド
        self.btn_Management = (592,387) # 管理
        self.btn_AllHello = (1162,777) # 一括挨拶
        self.btn_AllGet = (1346,777) # 一括受取

        self.btn_Challenge = (826,778) # 再挑戦
        self.btn_Next = (1378,777) # NEXT
        self.btn_Use = (1111,665) # 使用
        self.btn_OK = (832,704) # はい

        self.btn_Home = (416,225) # HOME       
        return

    def start(self):
        return

    def arena(self):
        x,y = self.btn_Arena
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(7)
        x,y = self.btn_OK_2
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(0.5)

        x,y = self.btn_ArenaReady
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(3)
        x,y = self.btn_Battle
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(120)
        pyautogui.click()
        time.sleep(5)
        x,y = self.btn_ArenaTop
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()

        time.sleep(4)
        x,y = self.btn_OK_2
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(2)
        x,y = self.btn_Home
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        return

    def quest_main(self):
        return
    def quest_sp(self):
        x,y = self.btn_Quest
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(3)
        x,y = self.btn_SpQuest
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(3)

        x,y = self.btn_EquipmentQuest
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(3)
        x,y = self.btn_EqQuest_Weapon
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(7)
        x,y = self.btn_Next
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(90)
        pyautogui.click()
        time.sleep(3)
        pyautogui.click()
        time.sleep(7)
        x,y = self.btn_Next
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(3)

        x,y = self.btn_QuestSelect
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(3)
        x,y = self.btn_Home
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        return
    def quest_guild(self):
        x,y = self.btn_Guild
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(3)
        x,y = self.btn_GuildQuest
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(3)
        x,y = self.btn_OK_2
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(0.5)
        x,y = self.btn_GuildQuestReady
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(2)
        x,y = self.btn_Battle
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(250)
        pyautogui.click()
        time.sleep(7)
        x,y = self.btn_Top
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(5)
        x,y = self.btn_Home
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        return
    def character(self):
        return
    def equip(self):
        return
    def get(self):
        return
    def friend(self):
        x,y = self.btn_Friend
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(3)
        x,y = self.btn_Management
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(3)
        x,y = self.btn_AllHello
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(3)
        x,y = self.btn_OK_2
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(3)
        x,y = self.btn_AllGet
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(3)
        x,y = self.btn_OK_2
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        time.sleep(3)
        x,y = self.btn_Home
        pyautogui.moveTo(x, y, 2, pyautogui.easeInBounce)
        pyautogui.click()
        return

def run():
    time.sleep(5)
    auto = AutoSet()
    #auto.arena()
    auto.quest_sp()
    auto.quest_guild()
    auto.friend()
    return

#getPosition()
run()