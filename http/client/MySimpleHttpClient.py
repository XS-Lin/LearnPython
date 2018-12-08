# スタブサーバに特定データPANを送信し、結果を表示する。
# スタブサーバはPANによって、結果を戻す

import urllib.request
import urllib.response
import urllib.parse

data = urllib.parse.urlencode({"PAN":"1234123412341234",'spam': 1, 'eggs': 2, 'bacon': 0})
data = data.encode('utf-8')
with urllib.request.urlopen("http://localhost:8000", data) as f:
    print(f.read().decode('utf-8'))
