'''
Created on Mar 6, 2018

@author: Justin Veyna
referemced:
https://stackoverflow.com/questions/12277933/send-data-from-a-textbox-into-flask
http://flask.pocoo.org/
'''
from flask import Flask, request, render_template
from synset_avg_generator import synset_entry
from free_input import sentence_to_output


app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    sentence = request.form["sentence"]
    processed_text = sentence_to_output(sentence)
    return processed_text