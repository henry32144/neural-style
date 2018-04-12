import os
from werkzeug import secure_filename
from flask import Flask, render_template, request, g , jsonify
from flask import send_from_directory, redirect, url_for
import handdraw, base64, json, sys
import transfer

## Declare allowed file extensions
ALLOWED_EXTENSIONS = set(['png', 'jpg','jpeg' ,'gif', 'PNG','JPG', 'JPEG','GIF'])

## Declare file path
STYLE_IMG_1_PATH = 'static/img/bamboo.jpg'
STYLE_IMG_2_PATH = 'static/img/misty-mood.jpg'
STYLE_IMG_3_PATH = 'static/img/wave.jpg'

## Initialize flask app.
app = Flask(__name__)

## This function check whether the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


## Render Index Page
@app.route('/')
def index():
    return render_template(
        'index.html')

## Deal with upload request
@app.route('/upload_image', methods=['POST'])
def upload_file():
    ## Get message from the form
    file = request.files['file']
    style = request.form['style']
    ## Print style to log
    print(style, file=sys.stdout)
    STYLE_IMG_PATH = None
    ## Check style
    if style == 'img-bamboo':
        STYLE_IMG_PATH = STYLE_IMG_1_PATH
    elif style == 'img-misty-mood':
        STYLE_IMG_PATH = STYLE_IMG_2_PATH
    else:
        STYLE_IMG_PATH = STYLE_IMG_3_PATH
    ## If file is valid
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        ## Start transfer
        data = transfer.transfer(file, STYLE_IMG_PATH, filename)
        ## encode image data to base64
        data = base64.b64encode(data).decode('UTF-8')
        ## put data into json
        json_data = json.dumps({'image':data})
        return json_data


if __name__ == '__main__':
    app.run(debug=True)
