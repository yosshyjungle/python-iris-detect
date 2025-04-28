from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('iris.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    try:
        data1 = request.form['sl']
        data2 = request.form['sw']
        data3 = request.form['pl']
        data4 = request.form['pw']
        arr = np.array([[data1, data2, data3, data4]])
        pred = model.predict(arr)
        return render_template('after.html', data=pred)
    except Exception as error:
        print(error)

if __name__ == '__main__':
    app.run()
