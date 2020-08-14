from sentiment_analysis import SentimentClassifier, lemmatize
from flask import Flask, render_template, request

app = Flask(__name__)


classifier = SentimentClassifier()
print("Classifier is ready")

#lemmatize = joblib.load("lemmatization.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods = ['GET', 'POST'])
def mra(text="", prediction_message=""):
    if request.method == 'POST':
        text = request.form.get("text")
        lem_text = lemmatize(text)
        #print(lem_text)
        prediction_message = classifier.get_prediction_message(lem_text)

    return render_template('index.html', text=text, prediction_message=prediction_message)


if __name__ == "__main__":
    app.run()