from django.contrib import admin

# Register your models here.
from .models import UserInfo


class UserInfoAdmin(admin.ModelAdmin):
    fileds = [
        'user_name',
        'mail_addr'
    ]

admin.site.register(UserInfo, UserInfoAdmin)