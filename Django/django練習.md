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

* front
  * bootstrap 4.3.1
  * jquary 3.4.1
* middleware
  * nginx 1.17.1
  * python3 3.7.4
    * uwsgi 2.0.18
    * django 2.2.3
* db
  * postgresql 11.4

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

主要目標: Django練習
サブ目標: Nginx,PostgreSql,CentOs,Bootstrap,PlantUMLの簡単使用

### 機能定義 ###

1. ブログ

   1. 管理者の機能

     * 管理画面へのログイン・ログアウト
     * ブログ名・ブログ説明の編集
     * 記事の投稿・編集・削除
     * カテゴリの追加・編集・削除
     * タグの追加・編集・削除
     * 閲覧者の機能

   1. 閲覧者の機能

     * 記事の一覧表示（公開日順・カテゴリごと・タグごと）
     * 記事の詳細表示
     * メニューボタンでカテゴリ・タグの一覧表示

1. 読書リスト

   1. 書籍の情報(JANコード、名称)を登録する

   1. 表示、非表示を設定

   1. 購入日、読破日を設定

   1. 読書感想記載(メモ帳機能)

      1. タイトル、本文を登録できること。(ここの登録は新規、更新、削除の意味)

      1. 本文の文字の色やフォントを指定できること。

      1. タイトルの一覧表示できること

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

1. 言語とタイムゾーンの設定

   ネットで「LANGUAGE_CODE = 'ja_JP'」という情報もあるが、「translation.E001」のエラーになる。

   ~~~python
   #LANGUAGE_CODE = 'en-us'
   LANGUAGE_CODE = 'ja'
   #TIME_ZONE = 'UTC'
   TIME_ZONE = 'Asia/Tokyo'
   ~~~

1. Bootstrap v4.3.1、jQuery 3.4.1

   開発段階は調査しやすいように開発版使用

1. djangoのMVC

   **一般的なMVCのViewはdjangoでtemlate、一般的なMVCのControllerはdjangoでviewと呼ぶ**

1. djangoのFormView,CreateView,UpdateView,DeleteView

   ドキュメント[Generic editing ビュー](https://docs.djangoproject.com/ja/2.2/ref/class-based-views/generic-editing/)

   1. HttpRequst処理の流れ(CreateViewのPOST処理)

      1. View.dispatch
      1. BaseCreateView.post
      1. ProcessFormView.post
      1. ProcessFormView.form_valid (チェックOKとする)
      1. ModelFormMixin.form_valid (OBJECT保存)
      1. FormMixin.form_valid
      1. HttpResponseRedirect
      1. HttpResponseRedirectBase
      1. HttpResponse

   1. HttpRequst処理の流れ(DetailViewのGET処理)

      1. View.dispatch
      1. BaseDetailView.get  (SingleObjectMixin.get_object)
      1. TemplateResponseMixin.render_to_response
      1. TemplateResponse
      1. SimpleTemplateResponse
      1. HttpResponse

   1. HttpRequst処理の流れ(UpdatelViewのGET処理)

      1. View.dispatch
      1. BaseUpdateView.get
      1. ProcessFormView.get
      1. ProcessFormView.get
      1. TemplateResponseMixin.render_to_response
      1. TemplateResponse
      1. SimpleTemplateResponse
      1. HttpResponse

   1. HttpRequst処理の流れ(UpdatelViewのPOST処理)

      1. View.dispatch
      1. BaseUpdateView.post
      1. ProcessFormView.post
      1. ProcessFormView.form_valid (チェックOKとする)
      1. ModelFormMixin.form_valid
      1. FormMixin.form_valid
      1. HttpResponseRedirect
      1. HttpResponseRedirectBase
      1. HttpResponse

   1. HttpRequst処理の流れ(FormViewのPOST処理)

      1. View.dispatch
      1. ProcessFormView.post
      1. ProcessFormView.form_valid (チェックOKとする)
      1. FormMixin.form_valid
      1. HttpResponseRedirect
      1. HttpResponseRedirectBase
      1. HttpResponse

1. djangoの起動流れ(py manager.py runserver)

   1. ManagementUtility.execute
      1. django.setup
      1. apps.populate (Python37\Lib\site-packages\django\apps\registry.py)
      1. AppConfig.create (Python37\Lib\site-packages\django\apps\config.py)
      1. settings.py の INSTALLED_APPS に記述のモジュールを動的ロード
   1. Command (Python37\Lib\site-packages\django\core\management\commands\runserver.py)
   1. BaseCommand.run_from_argv
   1. Command.execute
   1. BaseCommand.execute
   1. Command.handle
   1. Command.run (autoreloaderを使用しないとする)
   1. Command.inner_run
   1. run (Python37\Lib\site-packages\django\core\servers\basehttp.py)
      1. Command.get_handler (runのパラメータであれるhandler,requestオブジェクト作成元)
      1. get_internal_wsgi_application (Python37\Lib\site-packages\django\core\servers\basehttp.py)
      1. import_string (python37\lib\site-packages\django\utils\module_loading.py)
      1. WSGI_APPLICATION=project.wsgi.application (project/settings.py)
      1. get_wsgi_application (Python37\Lib\site-packages\django\core\wsgi.py)
      1. WSGIHandler.__init__ (middlewareロード)
         1. BaseHandler.load_middleware (Exceptionなしとする)
            1. _get_response (Python37\Lib\site-packages\django\core\handlers\base.py)
            1. ミドルウェアインスタンス作成(プロジェクトのsettingsのMIDDLEWAREで定義、実行順は定義順の逆)
            1. _view_middlewareにview関数を登録
         1. BaseHandler._get_response
         1. middleware_method (戻り値がなければ次を実行)
            1. _view_middlewareに登録されているview関数を実行(パラメータ:request, callback, callback_args, callback_kwargs)
         1. make_view_atomic (view関数を戻すして、次はrequestをパラメータとしてview関数を実行、view関数の戻り値はresponse)
      1. WSGIHandler.__call__
         1. WSGIRequest (requestオブジェクト)
         1. BaseHandler.get_response
         1. _middleware_chain (handler作成段階でsettings.MIDDLEWAREに記述したもの)
   1. HTTPServer.serve_forever (pythonの標準クラス、HTTP通信関連)

1. ミドルウェア

   1. django.contrib.messages.middleware.MessageMiddleware
      1. MiddlewareMixin.__call__ (BaseHandler.load_middlewareで実行 Python37\Lib\site-packages\django\core\handlers\base.py)
      1. MessageMiddleware.process_request
         1. django.contrib.messages.storage.default_storage
         1. TODO:settings.MESSAGE_STORAGEが未設定の場合はどうなる？
      1. MessageMiddleware.process_response
   1. django.contrib.auth.middleware.AuthenticationMiddleware
      1. MiddlewareMixin.__call__ (BaseHandler.load_middlewareで実行 Python37\Lib\site-packages\django\core\handlers\base.py)
      1. AuthenticationMiddleware.process_request
      1. AuthenticationMiddleware.process_response
   1. django.middleware.csrf.CsrfViewMiddleware
      1. MiddlewareMixin.__call__ (BaseHandler.load_middlewareで実行 Python37\Lib\site-packages\django\core\handlers\base.py)
      1. CsrfViewMiddleware.process_request
      1. CsrfViewMiddleware.process_response
   1. django.middleware.common.CommonMiddleware
      1. MiddlewareMixin.__call__ (BaseHandler.load_middlewareで実行 Python37\Lib\site-packages\django\core\handlers\base.py)
      1. CommonMiddleware.process_request
      1. CommonMiddleware.process_response
   1. django.contrib.sessions.middleware.SessionMiddleware
      1. MiddlewareMixin.__call__ (BaseHandler.load_middlewareで実行 Python37\Lib\site-packages\django\core\handlers\base.py)
      1. SessionMiddleware.process_request
      1. SessionMiddleware.process_response

1. urls.py
   1. RoutePattern
   1. URLPattern

      ~~~python
      # プロジェクトのurls.py
      urlpatterns = [
         path('', views.IndexView.as_view(), name='index'),
      ]
      ~~~

      ~~~python
      # django\urls\conf.py
      def _path(route, view, kwargs=None, name=None, Pattern=None):
          if isinstance(view, (list, tuple)):
              # For include(...) processing.
              pattern = Pattern(route, is_endpoint=False)
              urlconf_module, app_name, namespace = view
              return URLResolver(
                  pattern,
                  urlconf_module,
                  kwargs,
                  app_name=app_name,
                  namespace=namespace,
              )
          elif callable(view):
              pattern = Pattern(route, name=name, is_endpoint=True)
              return URLPattern(pattern, view, kwargs, name)
          else:
              raise TypeError('view must be a callable or a list/tuple in the case of include().')
      path = partial(_path, Pattern=RoutePattern)
      ~~~

1. View (Python37\Lib\site-packages\django\views\generic\base.py)
   1. as_view
      1. view関数を戻す
      1. view関数はdispatchを呼び出す
      1. dispatchはViewインスタンスのget、postメソッドを呼び出す

       ~~~python
       @classonlymethod
       def as_view(cls, **initkwargs):
           """Main entry point for a request-response process."""
           for key in initkwargs:
               if key in cls.http_method_names:
                   raise TypeError("You tried to pass in the %s method name as a "
                                   "keyword argument to %s(). Don't do that."
                                   % (key, cls.__name__))
               if not hasattr(cls, key):
                   raise TypeError("%s() received an invalid keyword %r. as_view "
                                   "only accepts arguments that are already "
                                   "attributes of the class." % (cls.__name__, key))

           def view(request, *args, **kwargs):
               self = cls(**initkwargs)
               if hasattr(self, 'get') and not hasattr(self, 'head'):
                   self.head = self.get
               self.setup(request, *args, **kwargs)
               if not hasattr(self, 'request'):
                   raise AttributeError(
                       "%s instance has no 'request' attribute. Did you override "
                       "setup() and forget to call super()?" % cls.__name__
                   )
               return self.dispatch(request, *args, **kwargs)
           view.view_class = cls
           view.view_initkwargs = initkwargs

           # take name and docstring from class
           update_wrapper(view, cls, updated=())
           # and possible attributes set by decorators
           # like csrf_exempt from dispatch
           update_wrapper(view, cls.dispatch, assigned=())
           return view
       ~~~
