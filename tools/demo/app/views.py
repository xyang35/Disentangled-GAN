from app import app
from flask import render_template

@app.route('/')
def root():
    return "Hello world!"

@app.route('/index/')
def index():
    return "Index!"

@app.route('/<string:page_name>/')
def render_static(page_name):
    return render_template('%s.html' % page_name)
