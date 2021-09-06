import numpy as np
import pandas as pd
from flask import Flask, url_for, redirect, request, render_template, jsonify
from joblib import load
from collections import Counter

# 3 klasy +1
ilLabel = 4

def getLogin(results):
    all = results.size
    print(all)

    quant = Counter(results)

    # sort by values
    # sortquant = sorted(quant.items(), key=lambda pair: pair[1], reverse=True)
    #print(sortquant[0])

    common = sorted(quant, key=quant.get, reverse=True)  #win label

    for item in common:
        #print(quant[item])  # this will give you the count of element eg. login
        percent =round(((quant[item]/all)*100), 2)
        #print(str(item) + ": " + str(percent) + "%")
        print(item + " : " + str(percent)+"%")

    return common[0]

def getLabel(results):
    all = results.size
    print(all)
    values = Counter(results)
    poss = []

    for i in range(1, ilLabel):
        # print(i)
        # print((values[i] / all)*100)
        val = round(((values[i] / all) * 100), 2)
        poss.append([i, val])
        print(str(i) + ": " + str(val) + "%")

    # poss
    max_val = max(poss, [1])  # key=lambda x: x[1]
    if max_val[1] > 30:
        return max_val[0]
    else:
        return 0


app = Flask(__name__)  # template_folder='../templates'

# Load ai model
model = load('model.joblib')


# loaded_pickle
@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict_api', methods=['POST', 'GET'])
def predict_api():
    if request.data:
        data = request.get_json()
        print(data)
        data2 = data['keys']

        data = pd.json_normalize(data2)
        prediction = model.predict(data)
        print(prediction)

        return jsonify(getLogin(prediction))

if __name__ == "__main__":
    app.run(debug=True)
