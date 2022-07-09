# データサイエンス勉強 #

## データサイエンス練習 ##

### datascience-notebook 環境 ###

~~~powershell
docker pull jupyter/datascience-notebook
docker run -it --rm -p 10000:8888 -v D:/Site/MyScript/python_test/data_science/datascience-notebook:/home/jovyan/work jupyter/datascience-notebook
# Tokenは毎回起動の時に変わため、コンソールから確認
# http://127.0.0.1:10000/lab?token=1caedcc41c1d585823742b29ce0cd91cd93601213a0e152c
~~~

## Spark練習 ##

### pyspark-notebook 環境 ###

~~~powershell
docker pull jupyter/pyspark-notebook
docker run -d -P --name notebook jupyter/all-spark-notebook
~~~

## Kubernetes練習 ##

### Kubernetes 環境 ###

~~~powershell
# Docker Desktop
kubectl version
~~~

### PlantUML ###

~~~powershell
docker pull plantuml/plantuml-server
# jettyまたはtomcat
docker run --rm -d -p 10001:8080 plantuml/plantuml-server:jetty
docker run --rm -d -p 10001:8080 plantuml/plantuml-server:tomcat
~~~

## Airflow練習 ##

~~~powershell
docker pull apache/airflow
~~~

## その他 ##

~~~txt
add image to readme.md 
![alt text](http://url/to/img.png)
![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)
~~~

## 参考資料 ##

* [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* [Dockerイメージの保存場所を変更する方法 docker desktop for windows](https://penguin-coffeebreak.com/archives/534)
* [docker run](http://docs.docker.jp/v19.03/engine/reference/commandline/run.html)
* [docker reference](https://docs.docker.com/reference/)
* [Jupyter Docker Stacks](https://jupyter-docker-stacks.readthedocs.io/en/latest/index.html)
* [Jupyter Docker Stacks Selecting an Image](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html)
* [plantuml](https://plantuml.com/ja/)