
import time
from pprint import pprint
from zapv2 import ZAPv2

#start zap in deamon mode: zap.exe -deamon

target = 'http://127.0.0.1:8080'
zap = ZAPv2(apikey='doofog4j8dkamn23ocqpdhjm34', proxies={'http': 'http://127.0.0.1:8180', 'https': 'http://127.0.0.1:8180'})
print('Accessing target %s' % target)

#zap.urlopen(target)
contextName = 'TestApi'
if contextName in zap.context.context_list:
    zap.context.remove_context(contextName)
contextId = zap.context.new_context(contextName)
print('Context Name=' + contextName + "Id=" + contextId)

zap.context.include_in_context(contextName, 'http://127.0.0.1:8080.*')
zap.context.include_in_context(contextName, 'http://localhost:8080.*')

zap.context.exclude_all_context_technologies(contextName)
zap.context.include_context_technologies(contextName,'OS.Linux')
zap.context.include_context_technologies(contextName,'Db.Oracle')
zap.context.include_context_technologies(contextName,'WS.Tomcat')

zap.authentication.set_authentication_method(contextId,'formBasedAuthentication','loginUrl=https://192.168.0.1/dologin.html' '&loginRequestData=username%3D%7B%25username%25%7D%26' 'password%3D%7B%25password%25%7D')
zap.authentication.set_logged_in_indicator(contextId, loggedinindicatorregex='Logged in')
zap.authentication.set_logged_out_indicator(contextId, 'Sorry, the username or password you entered is incorrect')

userName = 'TestUser1'
userId = zap.users.new_user(contextId, userName)
zap.users.set_authentication_credentials(contextId, userId, 'username=MyUserName&password=MySecretPassword')
zap.users.set_user_enabled(contextId, userId, True)
print('User Name=' + userName + "Id=" +userId)

#scanid = zap.spider.scan_as_user(contextId,userId,target)
# Give the Spider a chance to start
#time.sleep(2)
#while (int(zap.spider.status(scanid)) < 100):
#    print('Spider progress %: ' + zap.spider.status(scanid))
#    time.sleep(2)
#print('Spider completed')
# Give the passive scanner a chance to finish
#time.sleep(5)

scan_policy = "MyCustomScanPolicy"
zap.ascan.set_scanner_attack_strength(40018, "HIGH", scan_policy)
zap.ascan.set_scanner_attack_strength(0, "LOW", scan_policy)
#zap.ascan.scan(target, scanpolicyname = scan_policy)
#while (int(zap.ascan.status(scanid)) < 100):
#    print('Scan progress %: ' + zap.ascan.status(scanid))
#    time.sleep(5)
#print('Scan completed')

# Report the results
#print('Hosts: ' + ', '.join(zap.core.hosts))
#print('Alerts: ')
#pprint(zap.core.alerts())

#authentication
#https://stackoverflow.com/questions/44370019/owasp-zap-python-api-authentication
#session
#https://stackoverflow.com/questions/41123257/zap-api-session-authentication