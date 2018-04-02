import os
from werkzeug import secure_filename
from flask import Flask, render_template, request, g , jsonify
from flask import send_from_directory, redirect, url_for
import handdraw, base64, json, sys

OUTPUT_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

## Initialize flask app.
app = Flask(__name__)
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


## Render Index Page
@app.route('/')
def index():
    return render_template(
        'index.html')

@app.route('/upload_image', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        data = handdraw.imageprocess(file,filename)
        print(type(data), file=sys.stdout)
        data = base64.b64encode(data).decode('UTF-8')
        json_data = json.dumps({'image':data,'test':'no'})
        #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print(type(json_data), file=sys.stdout)
        return json_data

@app.route('/uploads/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'],
                               filename,as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
