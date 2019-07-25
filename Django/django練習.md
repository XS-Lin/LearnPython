# Django練習 #

## 仮想サーバ構成 ##

* ホスト: Windows 10 Home
* VMルール: Virtual Box 6
* ゲストOS: CentOs7.6
  * 仮想CPU: 2
  * メモリ: 4GB
  * ディスク: 40GB
  * ネットワーク　アダプタ: NAT + ホストオンリー

## 練習環境構成 ##

* python3
* postgresql
* django
* nginx

## 環境構築 ##

1. VM作成とゲストOSのインストール

   1. インストールオプション

      * 最小インストール

   1. ディスク構成

      * ディフォルト

   1. ユーザ作成

      * root/TestDjango1
      * dbadmin:webappadmin/postgre
      * appadmin:webappadmin/django
      * networkadmin:webappadmin/nginx

      ~~~sh
      groupadd -g 40000 webappadmin
      groupadd -g 41000 backupdba
      useradd -u 40001 -g webappadmin -G backupdba dbadmin
      useradd -u 40002 -g webappadmin networkadmin  
      useradd -u 40003 -g webappadmin appadmin
      passwd dbadmin
      passwd networkadmin
      passwd appadmin
      ~~~

   1. アップデート

      ~~~sh
      nmcli d
      nmcli c m enp0s3 connection.autoconnect yes
      yum update
      ~~~

1. python3インストール

   ~~~sh
   python --version # デフォルトバージョンが 2.7.5なので、最新版の3.7.4をインストール
   yum install gcc zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel openssl-devel tk-devel libffi-devel
   curl -o /usr/local/src/Python-3.7.4.tgz https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tgz
   cd /usr/local/src/
   tar xvfz Python-3.7.4.tgz
   cd Python-3.7.4
   ./configure -prefix=/usr/local/
   ./configure --enable-shared
   make altinstall
   ln -s /usr/local/bin/python3.7 /usr/bin/python3
   ln -s /usr/local/lib/libpython3.7m.so.1.0 /lib64/
   ln -s /usr/local/bin/pip3.7 /usr/bin/pip3
   python3 --version # 3.7.4
   ~~~

   ~~~sh
   # インターネットで以下のようにpythonをpython3にするという情報があるが、
   unlink /bin/python
   ln -s /usr/local/bin/python3.7 /usr/bin/python
   # yumがpython2.7依存のため動作できるなくなる。
   # 以下のように修復できる。
   ln -s /bin/python2 /bin/python
   ln -s /bin/python2 /usr/bin/python
   ~~~

   ~~~sh
   # pipバージョンアップ
   pip3 install --upgrade pip
   ~~~

1. postgresqlインストール

   注意事項とリリースバージョンを参照[PostgreSQL RPM](https://yum.postgresql.org/repopackages.php)

   ~~~sh
   yum list postgresql-server # デフォルトバージョンが9なので、最新版の11をインストール
   yum install epel-release
   yum install https://download.postgresql.org/pub/repos/yum/reporpms/EL-7-x86_64/pgdg-redhat-repo-latest.noarch.rpm
   yum install postgresql11-server
   /usr/pgsql-11/bin/postgres --version
   /usr/pgsql-11/bin/postgresql-11-setup initdb
   systemctl start postgresql-11
   su - postgres
   psql -l # 3行出力
   exit
   ~~~

1. djangoインストール

   ~~~sh
   pip3 install django
   python3 -m django --version
   ~~~

1. uwsgiインストール

   ~~~sh
   pip3 install uwsgi
   ~~~

1. nginxインストール

   [Nginx download](https://nginx.org/en/download.html)
   [Nginx Installation instructions RHEL/CentOS](https://nginx.org/en/linux_packages.html#RHEL-CentOS)

   ~~~sh
   yum list nginx # デフォルトバージョンが1.12.2なので、最新版の1.17.1をインストール
   yum install yum-utils
   vi /etc/yum.repos.d/nginx.repo
   # -------------------------START-------------------
   [nginx-stable]
   name=nginx stable repo
   baseurl=http://nginx.org/packages/centos/$releasever/$basearch/
   gpgcheck=1
   enabled=1
   gpgkey=https://nginx.org/keys/nginx_signing.key

   [nginx-mainline]
   name=nginx mainline repo
   baseurl=http://nginx.org/packages/mainline/centos/$releasever/$basearch/
   gpgcheck=1
   enabled=0
   gpgkey=https://nginx.org/keys/nginx_signing.key
   # ---------------------------END ------------------
   yum-config-manager --enable nginx-mainline
   yum install nginx
   nginx -v
   ~~~

## 開発環境(Windows10 + VSCode) ##

1. デバッグ環境設定

1. 自動デプロイスクリプト作成

## チュートリアル ##

[さぁ始めましょう](https://docs.djangoproject.com/ja/2.2/intro/)

### 問題点と解決策 ###

## 練習用サイト作成 ##

### 機能定義 ###

1. メモ

1. 読書リスト

### 高可用性設計 ###

1. バックアップとリカバリ

1. クラスターとロードバランシング

### 試験 ###

1. 自動機能試験(Selenium)

1. セキュリティ試験(ZAP)

1. 高可用性試験

# 情報 #

~~~cmd
# インストール
pip install django

# バージョン
python -m django --version

# 試験用サイト
# {python_home}\Python37\Lib\site-packages\django\bin\django-admin.py
django-admin.py startproject mysite

# 起動
manage.py runserver
manage.py runserver <port>
manage.py runserver <ip>:<port>

# runserver の自動リロード:ソース変更後再起動不要

# App作成
manage.py startapp polls

# データベース作成(SQLLITE)
manage.py migrate

# モデル変更
manage.py makemigrations polls

# テーブル作成用SQL生成
manage.py sqlmigrate polls 0001

# データベース作成(SQLLITE)
manage.py migrate

# ログインできるユーザーを作成(練習用)
manage.py createsuperuser
Username: admin
Email address: admin@example.com
Password: 123456
Password (again): 123456
Superuser created successfully.
~~~

## その他 ##

VsCodeでDjangoチュートリアルのエラー
**Class 'Question' has no 'objects' memberpylint(no-member)**

   ~~~cmd
   pip install pylint-django
   #user setting
   "python.linting.pylintArgs": [
        "--load-plugins=pylint_django"
   ],
   ~~~