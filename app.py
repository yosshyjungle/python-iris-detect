from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# モデルの読み込み部分を修正
try:
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iris.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"モデルの読み込みエラー: {e}")


@app.route("/")
def main():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def home():
    try:
        data1 = float(request.form["sl"])
        data2 = float(request.form["sw"])
        data3 = float(request.form["pl"])
        data4 = float(request.form["pw"])
        arr = np.array([[data1, data2, data3, data4]])
        pred = model.predict(arr)
        return render_template("after.html", data=pred)
    except Exception as error:
        print(error)
        return str(error), 400


if __name__ == "__main__":
    app.run()
