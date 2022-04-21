# Kubernetes #

## 参考資料 ##

* 入門Kubernetes (ISBN978-4-87311-840-6)
* Kubernetes完全ガイド (ISBN978-4-295-00979-5)
* [kubernetes docs home](https://kubernetes.io/ja/docs/home/)
* [Docker Reference documentation](https://docs.docker.com/reference/)

## メモ ##

### 入門Kubernetes ###

#### 1章 Kubernetes入門 ####

* immutability
* declarative configuration
* online self-healing system

#### 2章 コンテナの作成と起動 ####

### Kubernetes完全ガイド ###

#### 1章 Dockerの復習と「Hello,Kubernetes」 ####

##### Hello World #####

~~~go
package main

import "fmt"

func main() {
  fmt.Printf("Hello World\n")
}
~~~

~~~Dockerfile
FROM golang:1.14.1-alpine3.11
COPY ./source/main.go ./
RUN go build -o ./go-app ./main.go
USER nobody
ENTRYPOINT [ "./go-app" ]
~~~

~~~powershell
cd D:\Site\MyScript\python_test\data_science\kubernetes
docker image build -t sample-image:0.1 .
docker run --name hello_word -t sample-image:0.1
docker rm hello_word
docker rmi sample-image:0.1
~~~

