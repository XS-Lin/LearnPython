from django.urls import path

from . import views

app_name = 'app'
urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),
    path('note', views.NoteFormView.as_view(), name='new_note'),
    path('note_c', views.NoteCreate.as_view(), name='create_note'),
    path('note_r/<int:pk>', views.NoteDetail.as_view(), name='display_note'),
    path('note_u/<int:pk>', views.NoteUpdate.as_view(), name='update_note'),
    path('note_d/<int:pk>', views.NoteDelete.as_view(), name='deleter_note'),
]