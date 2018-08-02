import os
import xml.etree.ElementTree as et
import subprocess
import shutil

configFilePath = "config.xml"
configFilePath = "config_local.xml"
#configFilePath = r"D:\Site\MyScript\python_test\config_local.xml"
tree = et.ElementTree(file=configFilePath)
root = tree.getroot()
base_path_node = root.find(r"./settings/setting[@id='base_path']")
base_path = "" if base_path_node.text is None else base_path_node.text.strip()

temp_path_node = root.find(r"./settings/setting[@id='temp_folder']")
temp_path = "" if temp_path_node.text is None else temp_path_node.text.strip()

java_home_node = root.find(r"./settings/setting[@id='java_home']")
java_home = "" if java_home_node.text is None else java_home_node.text.strip()

easybuggyPath = os.path.join(base_path,"bin","easybuggy.jar")
javaPath = os.path.join(java_home,"bin","java.exe")

targetFolder = os.path.join(temp_path,"bin")
if not os.path.exists(targetFolder):
    os.makedirs(targetFolder)

targetJar = os.path.join(targetFolder,"easybuggy.jar")
if not os.path.exists(targetJar):
    shutil.copyfile(easybuggyPath, targetJar)

cmd = javaPath + " -jar " + targetJar

print(cmd)
subprocess.call(cmd)