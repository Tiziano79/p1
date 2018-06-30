import cv2
import uuid
import base64
import requests
import numpy as np
from keras.models import model_from_json, load_model
from keras.preprocessing import image
from flask import Blueprint, request, jsonify, Flask, render_template,Response
import dlib
from imutils import face_utils
import tensorflow as tf


app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
emotions = ["anger", "disgust", "fear", "happy", "neutral", "sadness", "surprise"]
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
FACE_CASCADE = cv2.CascadeClassifier("HAAR/haarcascade_frontalface_default.xml")

# load model
print("[INFO] load model...")
model = load_model("data/cnn6_500epAD256BS.model")
#problema quando eseguivo l'inferenza in un thread diverso rispetto a dove ho caricato il mio modello
graph = tf.get_default_graph()
emotion = ""

def obtain_images(request):
    '''
    All three routes below pass the image in the same way as one another.
    This function attempts to obtain the image, or it throws an error
    if the image cannot be obtained.
    '''

    if 'image_url' in request.args:
        image_url = request.args['image_url']

        response = requests.get(image_url)
        encoded_image_str = response.content

    elif 'image_buf' in request.files:
        image_buf = request.files['image_buf']  # <-- FileStorage object
        encoded_image_str = image_buf.read()

    elif 'image_base64' in request.args:
        image_base64 = request.args['image_base64']

        ext, image_str = image_base64.split(';base64,')

        encoded_image_str = base64.b64decode(image_str)


    encoded_image_buf = np.fromstring(encoded_image_str, dtype=np.uint8)
    decoded_image_bgr = cv2.imdecode(encoded_image_buf, cv2.IMREAD_COLOR)

    image_rgb = cv2.cvtColor(decoded_image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    annotate_image = (request.args.get('annotate_image', 'false').lower() == 'true')
    if annotate_image:
        annotated_rgb = np.copy(image_rgb)
    else:
        annotated_rgb = None
    crop_image = (request.args.get('crop_image', 'false').lower() == 'true')
    if crop_image:
        crop_faces = True
    else:
        crop_faces = False
    return image_rgb, image_gray, annotated_rgb, crop_faces


def prediction_emotion(image_gray):
    global emotion
    emotion = ""
    print("inizio predict.....")
    if image_gray is not None:
        print("img gray...")
        rects = detector(image_gray, 1)
        # faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        # print(faces) #locations of detected faces
        for rect in rects:
            # only 1 face
            if (len(rects)) == 1:
                # compute the bounding box of the face and draw it on the
                # frame
                (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
                if (bX > 0 and bY > 0 and bW > 0 and bH > 0):
                    detected_face = image_gray[int(bY):int(bY + bH), int(bX):int(bX + bW)]  # crop detected face
                    detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48
                    img_pixels = image.img_to_array(detected_face)
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_pixels /= 255  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
                    print("predict........................")
                    # per problema thread diverso da quello del modello
                    global graph
                    with graph.as_default():
                        predictions = model.predict(img_pixels, batch_size=1, verbose=1)  # store probabilities of 7 expressions
                        print("post predict...")
                        # find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:neutral, 5:sad, 6:surprise
                        max_index = np.argmax(predictions[0])
                        max = np.max(predictions[0])
                        emotion = emotions[max_index]
                        print (emotion, "____...___", max)




@app.route("/")
def init():
    return render_template('index.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    image_rgb, image_gray, annotated_rgb, crop_faces = obtain_images(request)
    prediction_emotion(image_gray)
    response = jsonify("true")
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response



def gen_result():
    no_face = "noface"
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + open('static/img/emoji/static/%s.png' % (emotion if(emotion != "")else no_face), 'rb').read() + b'\r\n')


# nel index2.html si mette url della risposta "{{ url_for('risultato')}}"
@app.route('/risultato')
def risultato():
    return Response(gen_result(), mimetype='multipart/x-mixed-replace; boundary= frame')



if __name__ == "__main__":
    app.run()
    #app.run(host='0.0.0.0')
    #app.run(host="0.0.0.0", port=5000)
