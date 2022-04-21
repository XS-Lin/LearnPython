# データサイエンス勉強 #

## データサイエンス練習 ##

### datascience-notebook 環境 ###

~~~powershell
docker pull jupyter/datascience-notebook
docker run -it --rm -p 10000:8888 -v D:/Site/MyScript/python_test/data_science/datascience-notebook:/home/jovyan/work jupyter/datascience-notebook
# Tokenは毎回起動の時に変わため、コンソールから確認
# http://127.0.0.1:10000/lab?token=412ca9be39eaf766798b439e511508cdc686bf3a427951d5
~~~

## Spark練習 ##

### pyspark-notebook 環境 ###

~~~powershell
docker pull jupyter/pyspark-notebook
docker run -d -P --name notebook jupyter/all-spark-notebook
~~~

## 参考資料 ##

* [Dockerイメージの保存場所を変更する方法 docker desktop for windows](https://penguin-coffeebreak.com/archives/534)
* [docker run](http://docs.docker.jp/v19.03/engine/reference/commandline/run.html)
* [docker reference](https://docs.docker.com/reference/)
* [Jupyter Docker Stacks](https://jupyter-docker-stacks.readthedocs.io/en/latest/index.html)
* [Jupyter Docker Stacks Selecting an Image](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html)
