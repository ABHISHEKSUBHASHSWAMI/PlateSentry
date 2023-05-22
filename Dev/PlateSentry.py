from flask import Flask, render_template, send_file,request, Response
import os
import base64
from ultralytics import YOLO
import cv2
from IPython.display import HTML

#model
model = YOLO('Model/plateSentry.pt')

#image function
def image_detection(image):
    if isinstance(image, str):
        image = cv2.imread(image)
    predict = model(image)[0].plot()
    return predict

#video function
def video_detection(video):
    cap = cv2.VideoCapture(video)
    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = "static/predict/output.mp4"
    
    # Define the codec for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create the video writer object
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Loop through the frames
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret:
            # Apply your model to the frame
            infer = model(frame)
            annotated = infer[0].plot()
            
            # Write the annotated frame to the output video
            out.write(annotated)
            
            # Display the resulting frame
            #cv2.imshow('Frame', annotated)
            
            # Press 'q' on the keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    
    # Release the video capture and writer objects
    cap.release()
    out.release()
    
    # Close all the frames
    cv2.destroyAllWindows()
    return output_path


#live feed
def generate_frames():
    # Create a VideoCapture object and set the camera source
    cap = cv2.VideoCapture(0)

    # Loop through the frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run your model inference on the frame
            # Modify this part with your own model code
            #frame = cv2.flip(frame, 1)
            # Annotate the frame with your model's results
            annotated_frame = model(frame,conf=0.4)[0].plot()  # Replace with your own annotation code

            # Encode the annotated frame as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame as multipart response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        else:
            break

    # Release the VideoCapture and close windows if any
    cap.release()
    cv2.destroyAllWindows()



#encode function
def encode_image(image_path):
    with open(image_path, "rb") as f:
        image_data = f.read()
        encoded_image = base64.b64encode(image_data).decode("utf-8")
        return encoded_image
    
def encode_video(video_path):
    with open(video_path, "rb") as f:
        video_data = f.read()
        encoded_video = base64.b64encode(video_data).decode("utf-8")
        return encoded_video

#extension  
ALLOWED_EXTENSIONS_IMAGE = {'png', 'jpg', 'jpeg'}
ALLOWED_EXTENSIONS_VIDEO = {'mp4','mov'}

def allowed_file_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_IMAGE

def allowed_file_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_VIDEO


#app

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
        if not allowed_file_image(image.filename):
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
        return render_template('imupload.html', upload=encode_image(image_path),result=encode_image('static/predict/output.jpg'))

    return render_template('imupload.html')


@app.route('/video', methods=['GET','POST'])
def video_page():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'video' not in request.files:
            return 'No file uploaded', 400

        video = request.files['video']
        if video.filename == '':
            return 'No file selected', 400
        
        # Check if the file has an allowed extension
        if not allowed_file_video(video.filename):
            return 'Invalid file type', 400
        
        # Save uploaded video to static/upload folder
        upload_folder = os.path.join(os.getcwd(), 'static', 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        vid_path = os.path.join(upload_folder, video.filename)
        video.save(vid_path)
        output_path=video_detection(vid_path)
        return render_template('video.html', video=encode_video(vid_path),output=encode_video(output_path))

    return render_template('video.html')


@app.route('/live', methods=['GET','POST'])
def live():
    if request.method == 'POST':
        return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

    return render_template('live.html')

@app.route('/yolo')
def paper():
    return send_file('docs/yolo.pdf', mimetype='application/pdf')

@app.route('/report')
def report():
    return send_file('docs/Report.pdf',mimetype='application/pdf')


if __name__ == '__main__':
    app.run(port=5748,debug=True)

