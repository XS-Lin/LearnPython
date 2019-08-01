from django.urls import path

from . import views

app_name = 'app'
urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),
    path('note', views.NoteView.as_view(), name='new_note'),
    path('note/<int:id>', views.NoteView.as_view(), name='curr_note'),
]