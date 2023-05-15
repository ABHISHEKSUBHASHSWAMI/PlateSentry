from flask import Flask, render_template, redirect, send_file

app = Flask('PlateSentry')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image')
def image_page():
    return render_template('image.html')

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
    app.run(debug=True)
