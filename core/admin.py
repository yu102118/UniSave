from django.contrib import admin
from .models import Document, Page, Chunk

# Это скажет Django: "Показывай эти таблицы в админ-панели"
admin.site.register(Document)
admin.site.register(Page)
admin.site.register(Chunk)