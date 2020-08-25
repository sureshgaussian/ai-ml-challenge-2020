from flask import Flask, render_template, request
from flask import redirect, url_for, make_response, jsonify
from datetime import datetime
import json
app = Flask(__name__)

#file extensions handled
app.config['UPLOAD_EXTENSIONS'] = ['.doc', '.pdf', '.docx']
app.config['MODEL_PATH'] = "../Inference"
app.config['COMMON_MODEL_PATH'] = "../common"

import os
import pandas as pd

# add the model path to the python path

import sys
sys.path.insert(1, app.config['MODEL_PATH'])
sys.path.insert(1, app.config['COMMON_MODEL_PATH'])
import parsers
import inference

fparsers = {}
fparsers['.doc'] = parsers.get_clauses_from_word
fparsers['.docx'] = parsers.get_clauses_from_word
fparsers['.pdf'] = parsers.get_clauses_from_pdf

def get_parser(filename):
    # We only want files with a . in the filename
    parser = None
    print("in get_parser function ")
    fname, ext = os.path.splitext(filename)
    # Check if the extension is in UPLOAD_EXTENSIONS
    if ext in app.config["UPLOAD_EXTENSIONS"]:
        parser = fparsers[ext]
    else:
        print(ext, " not in the approved list ")
    return parser

""" @app.route("/upload-file", methods=["GET", "POST"])
def upload_video():

    if request.method == "POST":

        file = request.files["file"]

        print("File uploaded")
        print(file)

        res = make_response(jsonify({"message": "File uploaded"}), 200)

        return res

    return render_template("upload_file.html")
 """
def get_bg_color(x):
    if x < 0.75:
        color = 'bg-warning'
    else:
        color = 'bg-light'
    return color

def get_recommendation(x):
    if x ==1:
        recommend = 'Reject'
    else:
        recommend  = 'Accept'
    return recommend


def get_text_color(x):
    if x == 1:
        color = 'text-danger'
    else:
        color = 'text-success'
    return color 

def add_colors(inferred):
    docdf = inferred['inference']
    docdf['text_color'] = docdf['label'].apply(get_text_color)
    docdf['bg_color'] = docdf['probability'].apply(get_bg_color)
    docdf['recommend'] = docdf['label'].apply(get_recommendation)    
    docdf['color' ] = docdf['text_color'] + " " + docdf['bg_color']
    docdf['id'] = docdf.index

    inferred['inference'] = docdf
    return inferred

@app.route("/analyze_doc", methods=['GET', 'POST'])
def analyze_doc():
    message = ''
    analysis_data = None
    file_name = ''

    # make sure input is of type doc, docx, or pdf. Otherwise, send an message
    if request.method == 'POST':
        if request.files:
            doc = request.files['document']
            file_name = doc.filename
            parser = get_parser(file_name)
            if parser:
                
                inferred = inference.infer(doc, parser)
                inferred = add_colors(inferred)

                analysis_data= inferred['inference'].to_dict('records')
                message = inferred.get('message', '')
            else:
                message = "Analysis for " + doc.filename + " is not supported. Only doc, docx, and pdf files supported"
            
        else:
            print("Empty file sent")
            message='No file selected to analyze'

    return render_template('new_home.html', analysis_data = analysis_data, message=message, doc_name = file_name, dont_show_filebar=True)

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route("/update_decisions", methods=['GET',  'POST'])
def update_decisions():
    return render_template('new_home.html')

@app.route("/test_analyze", methods=['GET', 'POST'])
def test_analyze():
    texts = ["wjlkajdlsfjaldsjflkjsldjflsdjflkajdfijlkflsadufosjflkadlfiusdfjl",
            " lklksdlwhat ever dyou kwhat is oging on ..",
            " lets see if this shows up properly in the text"]
    labels = [0,1,0]
    probability = [0.45, 0.66, 0.78]
    ids = [1,2,3]
    df = pd.DataFrame(list(zip(ids, texts, labels, probability)), columns=['id', 'Clause', 'label', 'probability'])
    df['recommend'] = df['label'].apply(get_recommendation)
    data  = df.to_dict('records')
    return render_template('new_home.html', analysis_data = data, message=None, doc_name = "abc.pdf")   

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template("new_home.html")
    #return redirect(url_for('upload_video'))

if __name__ == '__main__':

    app.run(host="192.168.1.6", debug=True)
