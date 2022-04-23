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

#### 3章 Kubernetesクラスターのデプロイ ####

~~~powershell
kubectl version
kubectl get componentstatuses
kubectl get nodes
kubectl describe nodes docker-desktop
~~~

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

#### 4章 APIリソースとkubectl ####

~~~yaml
apiVersion: v1
kind: Pod
metadata:
  name: sample-pod
spec:
  containers:
  - name: nginx-container
    image: nginx:1.16
~~~

~~~powershell
cd D:\Site\MyScript\python_test\data_science\kubernetes
kubectl create -f sample-pod.yaml
kubectl get pods

kubectl delete -f sample-pod.yaml
kubectl delete pod sample-pod

kubectl delete -f sample-pod.yaml --wait
kubectl delete -f sample-pod.yaml --grace-period 0 --force
~~~

~~~powershell
kubectl create -f sample-pod.yaml
kubectl apply -f sample-pod.yaml
kubectl get pod sample-pod -o jsonpath="{.spec.containers[?(@.name == 'nginx-container')].image}"
kubectl describe pod sample-pod
kubectl get pod sample-pod -o json
kubectl get pod sample-pod -o yaml

kubectl wait --for=condition=Ready pod/sample-pod
kubectl wait --for=condition=PodScheduled pod --all

kubectl annotate pods sample-pod annotations3=val3
kubectl get pod sample-pod -o jsonpath="{.metadata.annotations}"
kubectl annotate pods sample-pod annotations3-

kubectl label pods sample-pod label3=1234
kubectl get pod sample-pod -o jsonpath="{.metadata.labels}"
kubectl label pods sample-pod label3-

# label1=va1とlabel2を持つPod表示
kubectl get pods -l label1=va1,label2

kubectl get pods --show-labels

# 注意：直接設定変更の場合、手元のマニフェストファイルは更新されない
kubectl set image pod sample-pod nginx-container=nginx:1.15 
kubectl diff -f sample-pod.yaml

kubectl api-resources
kubectl api-resources --namespaced=true
kubectl get pods -o wide
kubectl get all

kubectl top node
kubectl -n kube-system top pod

kubectl exec -it sample-pod -- /bin/ls
kubectl exec -it sample-pod -c nginx-container -- /bin/ls
kubectl exec -it sample-pod -- /bin/bash -c "ls --all --classify | grep lib"

kubectl port-forward sample-pod 8888:80
curl http://localhost:8888/ # 別のターミナル

kubectl logs sample-pod
kubectl logs sample-pod -c nginx-container
kubectl logs --since=1h --tail=10 --timestamps=true sample-pod

kubectl cp
kubectl plugin list
~~~

~~~powershell
# podが起動しない場合のデバッグ
kubectl logs
kubectl describe # Events
kubectl run

kubectl run --iamge=nginx:1.16 --restart=Never --rm -it sample-debug --command -- /bin/sh
~~~

#### 5章 WorkloadsAPIsカテゴリ ####
