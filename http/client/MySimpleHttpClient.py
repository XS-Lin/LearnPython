# スタブサーバに特定データPANを送信し、結果を表示する。
# スタブサーバはPANによって、結果を戻す

import urllib.request
import urllib.parse

reqAddtionalHeaders = {
    'X-Request=Process':"01"
}

data = urllib.parse.urlencode({"PAN":"1234123412341234"})
data = data.encode('utf-8')

req = urllib.request.Request("http://localhost:8000", data,reqAddtionalHeaders)

with urllib.request.urlopen(req) as f:
    print(f.read().decode('utf-8'))
 