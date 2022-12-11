from flask import Flask
from flask import Flask, jsonify, request
from marshmallow import Schema, fields, ValidationError
from gpt_index import GPTTreeIndex, LLMPredictor
from langchain import OpenAI

app = Flask(__name__)

llm_predictor = LLMPredictor(
    llm=OpenAI(
        temperature=0,
        model_name="text-davinci-003",
    ),
)


playht_index = GPTTreeIndex.load_from_disk(
    'index.json',
    llm_predictor=llm_predictor,
)
twitter_index = GPTTreeIndex.load_from_disk(
    'twitter_index.json',
    llm_predictor=llm_predictor,
)


class BaseSchema(Schema):
    question = fields.String(required=True)


@app.route('/ask/playht', methods=["POST"])
def playht():
    # Get Request body from JSON
    request_data = request.json
    # Validate request body against schema data types
    schema = BaseSchema()
    try:
        result = schema.load(request_data)
    except ValidationError as err:
        return jsonify(err.messages), 400

    question = result['question']
    result = playht_index.query(question)

    # Send response
    return result, 200


@app.route('/ask/twitter', methods=["POST"])
def twitter():
    # Get Request body from JSON
    request_data = request.json
    # Validate request body against schema data types
    schema = BaseSchema()
    try:
        result = schema.load(request_data)
    except ValidationError as err:
        return jsonify(err.messages), 400

    question = result['question']
    result = twitter_index.query(question)

    # Send response
    return result, 200
