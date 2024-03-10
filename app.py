from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        hours = int (request.form['hours'])
        prev_scores = int (request.form['prev_scores'])
        sleep = int (request.form['sleep'])
        sample = int (request.form['sample'])

        input = np.array([[hours, prev_scores, sleep, sample]])
        prediction = model.predict(input)[0]

        return '<h1> %.2f %% is expected </h1>' % prediction[0]

if __name__ == '__main__':
    app.run(debug=True)