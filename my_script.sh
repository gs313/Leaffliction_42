#!/bin/bash

python3 -m venv leaves_venv
source leaves_venv/bin/activate
pip install --upgrade pip
pip install -r requirement.txt

exec $SHELL

# django-admin startproject mysite
# cd mysite
# python manage.py startapp helloworld


# pip freeze > requirement.txt