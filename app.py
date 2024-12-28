from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_sales():
    Store = request.form.get('Store')
    Department = request.form.get('Department')

    result = model.predict(np.array([Store, Department]))

    return str(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
