from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic

from .models import Note,BookInfo,BookNote

class IndexView(generic.ListView):
    template_name = 'index.html'
    context_object_name = 'message'
    
    def get_queryset(self):
        return "Test success!"

class NoteView():
    model = Note
    template_name = 'note_detail.html'

class BookNoteView():
    model = BookNote
    template_name = 'note_detail.html'

class BookInfoView():
    model = BookInfo
    template_name = 'book_detail.html'

def AddNote(request, book_note_id):
    return

def AddBookNote(request, book_note_id):
    return

