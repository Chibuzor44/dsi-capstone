import pickle
import pandas as pd
from util import display
from flask import Flask, request
from flask import render_template, flash, redirect, url_for
from pymongo import MongoClient
client = MongoClient()
app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Home')

@app.route('/predict', methods=['POST'] )
def predict():
    state = str(request.form['state'])
    bus_name = str(request.form['restaurant'])

    with open("../../../downloads/models.pickle", "rb") as f:
        models = pickle.load(f)
    db = client.yelp
    df = pd.DataFrame(list(db.review.find({"bus_name": bus_name, "state":state})))
    # with open("data.pickle", "wb") as f:
    #     pickle.dump(df, f)
    #
    # df.to_html("data.html")
    name, size, Recall, Precision, Accuracy, lst_neg, lst_pos = display(models, df, 4, 5, state=state, bus_name=bus_name)

    return render_template('predict.html', title='Home', name=bus_name, size=size, recall=Recall,
                           precision=Precision, acc=Accuracy, neg=dict(lst_neg), pos=lst_pos)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
