from flask import Flask

app = Flask(__name__)

classifier = None


@app.route("/")
def hello():
    return "Hello World!"

