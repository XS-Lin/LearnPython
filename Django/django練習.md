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

   ~~~sh
   # Option
   pip3 install django-richtextfield
   pip3 install psycopg2
   pip3 install django-crispy-forms # TODO
   pip3 install django-filter # TODO
   ~~~

1. uwsgiインストール

   ~~~sh
   pip3 install uwsgi
   ~~~

1. nginxインストール

   [Nginx download](https://nginx.org/en/download.html)
   [Nginx Installation instructions RHEL/CentOS](https://nginx.org/en/linux_packages.html#RHEL-CentOS)

   ~~~sh
   yum list nginx # デフォルトバージョンが1.12.2、最新版の1.17.1をインストール
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

## チュートリアル ##

公式チュートリアル[さぁ始めましょう](https://docs.djangoproject.com/ja/2.2/intro/)
非公式チュートリアル[汎用業務Webアプリを最速で作る](https://qiita.com/okoppe8/items/54eb105c9c94c0960f14)

### 問題点と解決策 ###

1. 「py manage.py runserver」反応なし、原因はmanage.pyの1行目

   ~~~python
   #!/usr/bin/env python
   ~~~

   Windowsの場合、該当行を削除
   Linuxの場合、以下のように変更

   ~~~python
   #!/usr/bin/python3
   ~~~

   該当事象は2019/03時点は発生しないが、2019/07時点で発生なので、Windowの更新が原因かもしれない。
   (Windows10 OSビルド18362.239、python 3.7.4、django 2.2.3 発生確認)

## 練習用サイト作成 ##

### 機能定義 ###

1. メモ帳

   1. タイトル、本文を登録できること。(ここの登録は新規、更新、削除の意味)

   1. 本文の文字の色やフォントを指定できること。

   1. タイトルの一覧表示できること

      1. タイトル順、更新時間順で並べる

      1. 行をクリックすると、詳細を表示

1. 読書リスト

   1. 書籍の情報(JANコード、名称)を登録する

   1. 表示、非表示を設定

   1. 購入日、読破日を設定

   1. 読書感想記載(メモ帳機能)

## 開発環境(Windows10 + VSCode) ##

1. デバッグ環境設定

   1. プロジェクト作成

      ~~~powershell
      django-admin startproject project
      py manage.py startapp app
      ~~~

   1. モデル作成

      ~~~powershell
      py manage.py makemigrations app
      py manage.py sqlmigrate app 0001 # 変更内容チェック
      py manage.py migrate app
      ~~~

      **DBの種類によって生成されたSQLが違う**

   1. 管理ユーザ作成(ローカル試験用)

      ~~~powershell
      py manage.py migrate # 初回のみ
      py manage.py createsuperuser
      Username: admin
      Email address: admin@example.com
      Password: 123456
      Password (again): 123456
      Superuser created successfully.
      ~~~

   1. 管理ページアクセステスト

      ~~~powershell
      py manage.py runserver
      # browser http://127.0.0.1:8000/admin/
      ~~~

1. 自動デプロイスクリプト作成

### 高可用性設計 ###

1. バックアップとリカバリ

   1. 差分増分バックアップ(毎日)

   1. バックアップセットに適用(日曜日)

1. クラスターとロードバランシング

### 試験 ###

1. 自動機能試験(Selenium)

1. セキュリティ試験(ZAP)

1. 高可用性試験

## 情報 ##

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

## メモ ##

1. リクエスト プロセス フロー

   Request ->  RequsetMiddleware(urls.py) -> View(views.py) -> Model(models.py) -> DB(settings.py) -> Model -> View -> Response

1. URL(urls.py)

   ~~~python
   path(route,view,kwargs,name)
   # route:  https://www.example.com/myapp/?page=3 の場合、myapp/
   #   '<int：name_key>/' のように、name_keyでurlの一部をパラメータとして取得可能(htmlテンプレートの{%url%}で使用)
   # view: 呼び出すView関数設定
   #   views.fucntion_name で views.py に定義したfucntion_name関数設定
   # kwargs: TODO 後で調べる
   # name: TODO 後で調べる
   ~~~

1. View(views.py)

   view関数定義とhtml定義[引用元](https://docs.djangoproject.com/ja/2.2/intro/tutorial03/)

   ~~~python
   from django.http import HttpResponse
   from django.template import loader
   from .models import Question
   def index(request):
       latest_question_list = Question.objects.order_by('-pub_date')[:5]
       template = loader.get_template('polls/index.html')
       context = {
           'latest_question_list': latest_question_list,
           }
       return HttpResponse(template.render(context, request))
   def vote(request, question_id):
       question = get_object_or_404(Question, pk=question_id)
       try:
           selected_choice = question.choice_set.get(pk=request.POST['choice'])
       except (KeyError, Choice.DoesNotExist):
           return render(request, 'polls/detail.html', {
               'question': question,
               'error_message': "You didn't select a choice.",
           })
       else:
           selected_choice.votes += 1
           selected_choice.save()
           return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
   ~~~

   ~~~html
   {% if latest_question_list %}
       <ul>
       {% for question in latest_question_list %}
           <li><a href="{% url 'polls:detail' question.id %}">{{ question.question_text }}</a></li>
       {% endfor %}
       </ul>
   {% else %}
       <p>No polls are available.</p>
   {% endif %}
   ~~~

   ~~~html
   <h1>{{ question.question_text }}</h1>
   {% if error_message %}<p><strong>{{ error_message }}</strong></p>{% endif %}
   <form action="{% url 'polls:vote' question.id %}" method="post">
   {% csrf_token %}
   {% for choice in question.choice_set.all %}
       <input type="radio" name="choice" id="choice{{ forloop.counter }}" value="{{ choice.id }}">
       <label for="choice{{ forloop.counter }}">{{ choice.choice_text }}</label><br>
   {% endfor %}
   <input type="submit" value="Vote">
   </form>
   ~~~

   ~~~python
   render(request, template_name, context=None, content_type=None, status=None, using=None)
   # HttpResponse
   get_object_or_404(klass, *args, **kwargs)
   # modelobject
   ~~~

1. Model(models.py)

   ~~~python
   class ModelName(models.Model):
      modelfield = models.type(constraint)
   # 一般的なエンティティのイメージ
   makemigrations
   # 0001_initial.py にsql文生成できる。
   ~~~

1. DB(settings.py)

   ~~~python
   DATABASES = {
       'default': {
           'ENGINE': 'django.db.backends.sqlite3',
           'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
       }
   }
   # ENGINE: 'django.db.backends.sqlite3','django.db.backends.postgresql','django.db.backends.mysql','django.db.backends.oracle'
   # NAME: database name
   ~~~

1. VsCodeでDjangoチュートリアルのエラー

   **Class 'Question' has no 'objects' memberpylint(no-member)**

   ~~~cmd
   pip install pylint-django
   #user setting
   "python.linting.pylintArgs": [
        "--load-plugins=pylint_django"
   ],
   ~~~
