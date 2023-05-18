from flask import Flask, render_template, send_file,request
import os
import base64
from ultralytics import YOLO
import cv2
from platescript import ocr
import re


#model
model = YOLO('Model/plateSentry.pt')

def image_detection(image):
    if isinstance(image, str):
        image = cv2.imread(image)
    predict = model(image)[0].plot()
    return predict

def encode_image(image_path):
    with open(image_path, "rb") as f:
        image_data = f.read()
        encoded_image = base64.b64encode(image_data).decode("utf-8")
        return encoded_image
    
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask('PlateSentry')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/imupload', methods=['GET', 'POST'])
def image_upload():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'image' not in request.files:
            return 'No file uploaded', 400

        image = request.files['image']
        if image.filename == '':
            return 'No file selected', 400
        
         # Check if the file has an allowed extension
        if not allowed_file(image.filename):
            return 'Invalid file type', 400
        
        
        # Save uploaded image to static/upload folder
        upload_folder = os.path.join(os.getcwd(), 'static', 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        image_path = os.path.join(upload_folder, image.filename)
        image.save(image_path)

        # Apply image detection on the uploaded image
        predict = image_detection(image_path)
        cv2.imwrite('static/predict/output.jpg',predict)
        text=ocr(image_path)
        return render_template('imupload.html', upload=encode_image(image_path),result=encode_image('static/predict/output.jpg'),data=text)

    return render_template('imupload.html')



@app.route('/video')
def video_page():
    return render_template('video.html')

@app.route('/live')
def live_page():
    return render_template('live.html')

@app.route('/yolo')
def paper():
    return send_file('docs/yolo.pdf', mimetype='application/pdf')

@app.route('/report')
def report():
    pass


if __name__ == '__main__':
    app.run(port=5748,debug=True)

