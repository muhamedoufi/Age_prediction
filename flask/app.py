import os 
from flask import Flask, render_template, jsonify, request
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

# app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())



@app.route('/age_prediction', methods=["GET", "POST"])
def Age_prediction():

    if request.method == 'POST':
        img = request.form['question']
        test_pic = cv2.imread()
        image = cv2.cvtColor(test_pic,cv2.COLOR_BGR2RGB)
        test_pic = cv2.resize(image,(128,128))
        test_pic = test_pic.reshape((1,128,128,3))
        model = load_model('Age_predict_ml.h5')
        pred = model.predict(test_pic)

    return jsonify({"response": int(pred) })



if __name__ == '__main__':
    app.run(host='127.0.0.1', port='8888', debug=True)
