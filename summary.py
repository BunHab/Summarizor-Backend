from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

@app.route('/summary', methods=['POST'])
def summarize():
    data = request.get_json()
    article = data['article']

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    result = summarizer(article, max_length=130, min_length=30, do_sample=False)

    summary_text = result[0]['summary_text']

    return jsonify(result=summary_text)

if __name__ == '__main__':
    app.run()
