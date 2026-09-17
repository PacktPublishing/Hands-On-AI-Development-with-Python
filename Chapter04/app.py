from flask import Flask, render_template
from flask import request
import joblib

app = Flask(__name__)

model = joblib.load("imdb_sentiment_model.pkl")
vectorizer = joblib.load("imdb_vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form.get('review')
    if not review:
        return "No review available"
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)
    
    sentiment = "positive" if prediction[0] == 1 else "negative"
    return f"The movie review is <b>{sentiment}</b>"

if __name__ == '__main__':
    app.run(debug=True)
