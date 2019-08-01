from django.db import models

class UserInfo(models.Model):
    user_name = models.CharField(max_length=60)
    mail_addr = models.CharField(max_length=200)
    create_date = models.DateTimeField(auto_now_add=True,null=True)
    update_date = models.DateTimeField(auto_now=True,null=True)
    def __str__(self):
        return self.user_name

class Note(models.Model):
    title = models.CharField(max_length=60)
    content = models.TextField(blank=True)
    owner = models.ForeignKey(UserInfo,on_delete=models.DO_NOTHING,null=True)
    create_date = models.DateTimeField(auto_now_add=True,null=True)
    update_date = models.DateTimeField(auto_now=True,null=True)
    
    def __str__(self):
        return self.title

class BookInfo(models.Model):
    jan_code = models.CharField(max_length=50)
    title = models.CharField(max_length=300)
    is_display = models.BooleanField(default=True)
    create_date = models.DateTimeField(auto_now_add=True,null=True)
    update_date = models.DateTimeField(auto_now=True,null=True)

    def __str__(self):
        return self.title

class BookNote(models.Model):
    id = models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')
    book_id = models.ForeignKey(BookInfo,on_delete=models.DO_NOTHING,null=True)
    note_id = models.ForeignKey(Note,on_delete=models.CASCADE,null=True)
    read_date = models.DateTimeField()
    create_user = models.IntegerField(null=True)
    create_date = models.DateTimeField(auto_now_add=True,null=True)
    update_user = models.IntegerField(null=True)
    update_date = models.DateTimeField(auto_now=True,null=True)

    def __str__(self):
        return self.id