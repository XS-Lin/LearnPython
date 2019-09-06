from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic
from django.contrib import messages
from django.urls import reverse_lazy

from .models import Note,BookInfo,BookNote,BlogContent,Category,Tag
from .forms import NoteForm,BlogContentForm

class IndexView(generic.ListView):
    template_name = 'index.html'
    context_object_name = 'message'
    
    def get_queryset(self):
        return "Test success!"

class NoteFormView(generic.FormView):
    model = Note
    template_name = 'note_form.html'
    form_class = NoteForm
    success_url = 'note/'
    def form_valid(self, form):
        note = Note(title=form.cleaned_data['title'],content=form.cleaned_data['content'])
        note.save()
        self.success_url += str(note.id)
        return super().form_valid(form)

class NoteCreate(generic.CreateView):
    model = Note
    form_class = NoteForm
    template_name = 'note_form.html'
    
    def form_valid(self, form):
        self.success_url += reverse_lazy('note_r/',args=(self.object.id ))
        return super().form_valid(form)

class NoteDetail(generic.DetailView):
    model = Note
    form_class = NoteForm
    template_name = 'note_form.html'
    context_object_name = 'form'

class NoteUpdate(generic.UpdateView):
    model = Note
    form_class = NoteForm
    template_name = 'note_form.html'
    def form_valid(self, form):
        self.success_url = reverse_lazy('note_r/' + self.object.id)
        return super().form_valid(form)

class NoteDelete(generic.DeleteView):
    model = Note
    form_class = NoteForm
    template_name = 'note_form.html'

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
class BlogCreateView(generic.CreateView):
    template_name = 'blog.html'
    form_class = BlogContentForm

class BlogUpdateView(generic.UpdateView):
    template_name = 'blog.html'
    form_class = BlogContentForm

class BlogDetailView(generic.DetailView):
    template_name = 'blog.html'
    form_class = BlogContentForm