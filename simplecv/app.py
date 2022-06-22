from flask import Flask, render_template, request, redirect
from keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

model = load_model('final_model.h5')

model.make_predict_function()


def predictImage(img_path):
    img1 = tf.keras.utils.load_img(img_path, target_size=(150, 150))
    plt.imshow(img1)
    Y = tf.keras.utils.img_to_array(img1)
    X = np.expand_dims(Y, axis=0)
    val = model.predict(X)
    if val == 1:
        # return plt.xlabel("Пёсика", fontsize=30)
        return "Пёсика"
    elif val == 0:
        # return plt.xlabel("Котика", fontsize=30)
        return "Котика"
    return (val)


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/predicter", methods=['GET', 'POST'])
def get_output():
    global img_path, p
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/uploads/" + img.filename
        img.save(img_path)
        p = predictImage(img_path)
    if request.method == 'GET':
        return redirect("/")
    return render_template("index.html", prediction=p, img_path=img_path)


if __name__ == '__main__':
    app.run(debug=True)
