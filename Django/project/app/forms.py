from django.core import validators
from django import forms
from .models import Note

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