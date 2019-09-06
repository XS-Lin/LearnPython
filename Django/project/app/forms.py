from django.core import validators
from django import forms
from .models import Note,BlogContent

class NoteForm(forms.ModelForm):
    class Meta:
        model = Note
        fields = ('title','content')
        widgets = {
            'title':forms.TextInput(),
            'content': forms.Textarea(),
        }

    def clean_title(self):
        data = self.cleaned_data['title']
        if "a" in data:
            raise forms.ValidationError("Title cannot contains a")
        return data

    def clean(self):
        cleaned_data = super().clean()
        title = cleaned_data.get("title")
        content = cleaned_data.get("content")
        if title == content:
            raise forms.ValidationError("Content must be different from title.")

class BlogContentForm(forms.ModelForm):
    class Meta:
        model = BlogContent
        fields = ('category','title','content','comment','public_open','public_time')
        widgets = {
            'category':forms.TextInput(),
            'title':forms.TextInput(),
            'content': forms.Textarea(),
            'comment': forms.Textarea(),
            'public_open': forms.CheckboxInput(),
            'public_time': forms.DateTimeInput()
        }
    
    def clean_title(self):
        data = self.cleaned_data['title']
        if "a" in data:
            raise forms.ValidationError("Title cannot contains a")
        return data

    def clean(self):
        cleaned_data = super().clean()
        title = cleaned_data.get("title")
        content = cleaned_data.get("content")
        if title == content:
            raise forms.ValidationError("Content must be different from title.")