import os

import pandas as pd
from redis import Redis
from algorithm.utilitis import file_md5
from algorithm.ml_preds import LGBMPredictor
from flask import Flask, request, render_template
from flask import redirect, url_for

app = Flask(__name__)  # 申明app对象
redis = Redis(host='127.0.0.1', port=6379)  # host:数据库的服务名


@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route("/hide")
def admin():
    return redirect(url_for("login"))  # 回到登陆页面


@app.route('/redis')
def hello():
    redis.incr('hits')
    count = redis.get('hits').decode('utf-8')
    return f'Hello! I have been seen {count} times.'


@app.route('/', methods=['POST', 'GET'])
def upload():
    kind = request.form.get('kind')
    al = request.form.get('al')
    file = request.files.get('file')
    if not file:
        return render_template('index.html', message='No file selected!')
    filename = file.filename
    # file.save(os.path.join("upload", filename))
    return redirect(url_for('success'))


@app.route('/success')
def success():
    return 'File uploaded successfully.'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999, debug=True)
