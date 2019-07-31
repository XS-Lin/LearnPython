from django.db import models

class Note(models.Model):
    title = models.CharField(max_length=60)
    context = models.TextField(blank=True)
    create_date = models.DateTimeField(auto_now_add=True,null=True)
    update_date = models.DateTimeField(auto_now=True,null=True)
    
    def __str__(self):
        return self.title

class BookInfo(models.Model):
    jan_code = models.CharField(max_length=50)
    title = models.CharField(max_length=300)
    is_display = models.BooleanField(default=True)
    note_id = models.ForeignKey(Note,on_delete=models.CASCADE,null=True)
    create_date = models.DateTimeField(auto_now_add=True,null=True)
    update_date = models.DateTimeField(auto_now=True,null=True)

    def __str__(self):
        return self.title
    