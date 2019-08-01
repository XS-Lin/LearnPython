from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic
from django.contrib import messages

from .models import Note,BookInfo,BookNote
from .forms import NoteForm

class IndexView(generic.ListView):
    template_name = 'index.html'
    context_object_name = 'message'
    
    def get_queryset(self):
        return "Test success!"

class NoteView(generic.FormView):
    model = Note
    template_name = 'note_detail.html'
    form_class = NoteForm
    success_url = 'note/'
    def form_valid(self, form):
        form.model.save()
        success_url += form.model.id
        return super().form_valid(form)
    
class BookNoteView(generic.DetailView):
    template_name = 'note_detail.html'
    context_object_name = 'book_note'

class BookInfoView(generic.DetailView):
    model = BookInfo
    template_name = 'book_detail.html'

def AddNote(request, book_note_id):
    return

def AddBookNote(request, book_note_id):

    return

