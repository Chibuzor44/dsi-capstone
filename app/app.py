import pickle
# from fraud_model import MyModel, get_data
from flask import Flask, request
from flask import render_template, flash, redirect, url_for
app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Home')

@app.route('/predict', methods=['POST'] )
def predict():
    # X = str(request.form['user_input'])
    # with open('model.pkl', 'rb') as f:
    #     model = pickle.load(f)
    #
    # pred = "Predictions: {}".format(" ".join(model.predict([X])))
    return render_template('predict.html', title='Home')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
