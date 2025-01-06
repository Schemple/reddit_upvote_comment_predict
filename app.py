from flask import Flask, render_template, request, url_for
from predict import predict_voteups_api, predict_comments_api
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/aboutus' )
def aboutus():
    return render_template('aboutus.html')

@app.route('/predict', methods=['POST'])
def predict():
    subscribers = int(request.form['subscribers'])
    posting_time = datetime.strptime(request.form['posting_time'], '%Y-%m-%d %H:%M:%S')
    posting_title = request.form['posting_title']
    posting_content = request.form['posting_content']
    referenced_url = request.form['referenced_url']

    voteups = predict_voteups_api(subscribers, posting_time, posting_title, posting_content, referenced_url)
    comments = predict_comments_api(subscribers, posting_time, posting_title, posting_content, referenced_url)

    return render_template('result.html', voteups=voteups, comments=comments)

if __name__ == '__main__':
    app.run(debug=True)