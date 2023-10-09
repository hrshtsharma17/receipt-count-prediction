import re
from flask import Flask, render_template, request
from model.inference import Inference

DATA_PATH = "data/data_daily.csv"
MODEL_PATH = "model/artifacts_2023-10-08_20-11-39/best_model.h5"
MONTH_FORMAT = r'^\d{2}-\d{4}$'

app = Flask(__name__)
inf = Inference(MODEL_PATH, DATA_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    date = request.form['date']
    if not re.search(MONTH_FORMAT, date):
        out = "Please enter a valid month value (MM-YYYY)"
    else:
        out = inf.predict_by_month(date)
        if type(out)!=str:
            out = int(out)

    return render_template('index.html', data=out)

if __name__ == '__main__':
    app.run(host='localhost', port=3000)
