
import time
from pprint import pprint
from zapv2 import ZAPv2

target = 'http://127.0.0.1:8080'
zap = ZAPv2(apikey='doofog4j8dkamn23ocqpdhjm34', \
            proxies={'http': 'http://127.0.0.1:8180', 'https': 'http://127.0.0.1:8180'})
print('Accessing target %s' % target)

#zap.urlopen(target)

#time.sleep(2)
scanid = zap.spider.scan(target)
# Give the Spider a chance to start
time.sleep(2)
while (int(zap.spider.status(scanid)) < 100):
    print('Spider progress %: ' + zap.spider.status(scanid))
    time.sleep(2)

print('Spider completed')
# Give the passive scanner a chance to finish
time.sleep(5)

print('Scanning target %s' % target)
scanid = zap.ascan.scan(target)
while (int(zap.ascan.status(scanid)) < 100):
    print('Scan progress %: ' + zap.ascan.status(scanid))
    time.sleep(5)

print('Scan completed')

# Report the results

print('Hosts: ' + ', '.join(zap.core.hosts))
print('Alerts: ')
pprint(zap.core.alerts())