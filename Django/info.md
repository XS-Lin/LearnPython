
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