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

1. python3インストール

1. postgresqlインストール

1. djangoインストール

1. nginxインストール

## 開発環境(Windows10 + VSCode) ##

1. デバッグ環境設定

1. 自動デプロイスクリプト作成

## チュートリアル ##

[さぁ始めましょう](https://docs.djangoproject.com/ja/2.2/intro/)

### 注意すべき点(個人観点) ###

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