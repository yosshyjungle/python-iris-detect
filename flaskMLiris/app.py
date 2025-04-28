from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('iris.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def detect():
    try:
        data1 = request.form['sl']
        data2 = request.form['sw']
        data3 = request.form['pl']
        data4 = request.form['pw']
        arr = np.array([[data1, data2, data3, data4]])
        pred = model.predict(arr)
        return render_template(
            'results.html',
            data = pred
        )
    except Exception as error:
        print(error)

if __name__ == '__main__':
    app.run()

