from flask import Flask, render_template, Response, request
import datetime
import cv2
import os
from threading import Thread
import subprocess
import time

app = Flask(__name__)

camera = cv2.VideoCapture(0)
process_running = False

global capture, rec_frame, switch, rec, out
capture = 0
grey = 0
neg = 0
face = 0
switch = 1
rec = 0
name = ''

app.app_context().push()


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/front')
def front():
    return render_template('front.html')


@app.route('/logic')
def logic():
    try:
        result = subprocess.check_output(['python','compare.py'], text=True)
        return render_template('front.html')
    except subprocess.CalledProcessError as e:
        print(f"Error executing script: {e}")
        return render_template('error.html', error_message=str(e))


def FrameCapture(path):
    print(path)
    vidObj = cv2.VideoCapture(path)
    frames = vidObj.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vidObj.get(cv2.CAP_PROP_FPS)
    seconds = round(frames / fps)
    count = 0
    success = 1

    while seconds:
        if vidObj.read():
            success, image = vidObj.read()
        else:
            success = 0
            break

        print(success)
        folder_path = 'frame'
        os.makedirs(folder_path, exist_ok=True)
        cv2.imwrite(os.path.join(folder_path, f"frame{count}.jpg"), image)

        count += 1
        seconds = seconds - 1
        print(seconds)


try:
    os.mkdir('./shots')
except OSError as error:
    pass

def record(out, delay_seconds=5):
    global rec_frame
    time.sleep(delay_seconds)
    while rec:
        time.sleep(0.05)
        out.write(rec_frame)


def gen_frames():
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            if capture:
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', f"shot_{str(now).replace(':', '')}.png"])
                cv2.imwrite(p, frame)

            if rec:
                rec_frame = frame
                frame = cv2.putText(cv2.flip(frame, 1), "Recording...", (0, 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (76, 175, 80), 4)
                frame = cv2.flip(frame, 1)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera, name
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
        elif request.form.get('stop') == 'Stop/Start':
            if switch == 1:
                switch = 0
                camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get('change') == 'process':
            FrameCapture(name)
        elif request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec = not rec
            if rec:
                now = datetime.datetime.now()
                name = f'vid_{str(now).replace(":", "")}.avi'
                print(name)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(f'vid_{str(now).replace(":", "")}.avi', fourcc, 20.0, (640, 480))
                thread = Thread(target=record, args=[out, 5])  
                thread.start()
            elif not rec:
                out.release()

    elif request.method == 'GET':
        return render_template('front.html')
    return render_template('front.html')


if __name__ == '__main__':
    app.run(debug=True)
