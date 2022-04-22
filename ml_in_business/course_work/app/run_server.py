import pandas as pd
import dill
from flask import Flask, request, jsonify

with open('models/lr_pipeline.dill', 'rb') as in_strm:
    model = dill.load(in_strm)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def general():
    return "Welcome to prediction process"


@app.route('/predict', methods=['POST'])
def predict_insult():
    data = {"success": False}
    ident, comment_text = "", ""

    request_json = request.get_json()

    if request_json["ident"]:
        ident = request_json['ident']

    if request_json["comment_text"]:
        comment_text = request_json['comment_text']

    example = pd.DataFrame({'id': ident, 'comment_text': comment_text}, index=[0])
    preds = model.predict_proba(example)
    data['predictions'] = str(preds[:, 1][0])
    data['succes'] = True
    print('OK')

    return jsonify(data)


if __name__ == '__main__':
    app.run()
