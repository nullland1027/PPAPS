import os

import pandas as pd
from redis import Redis
from algorithm.utilitis import file_md5
from algorithm.ml_preds import LGBMPredictor
from flask import Flask, request, render_template
from flask import redirect, url_for

app = Flask(__name__)  # 申明app对象
redis = Redis(host='127.0.0.1', port=6379)  # host:数据库的服务名

# Global variables area
IS_LOGIN = False  # 用户是否登陆


@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def index():
    """The website cover page"""
    return render_template('index.html')


@app.route('/check-state')
def check_if_login():
    """Check"""
    if IS_LOGIN:
        return redirect(url_for("home"))
    else:
        return redirect(url_for("login"))


@app.route('/login', methods=['POST', 'GET'])
def login():
    global IS_LOGIN
    if IS_LOGIN:
        return redirect(url_for('home'))
    if request.method == 'POST':  # received user's input
        user_id = request.form['email']
        pwd = request.form['password']
        if user_id == 'admin@super.com' and pwd == '123':
            IS_LOGIN = True
            return redirect(url_for("home"))
    else:  # not receive user's input 由其他页面跳转而来
        return render_template('login.html')


@app.route('/signup', methods=['POST', 'GET'])
def sign_up():
    return render_template('sign-up.html')


@app.route('/home-page', methods=['POST', 'GET'])
def home():
    """After click"""
    global IS_LOGIN
    if not IS_LOGIN:  # 没有登陆
        return redirect(url_for("login"))

    kind = request.form.get('kind')
    al = request.form.get('al')
    file = request.files.get('file')

    if not file:
        return render_template('home-page.html')  # , message='No file selected!')
    filename = file.filename
    # file.save(os.path.join("upload", filename))
    return redirect(url_for('file_checking', kind=kind, al=al))


@app.route('/file-checking')
def file_checking():
    """The file has been uploaded"""
    kind = request.args.get('kind')
    al = request.args.get('al')
    return f'File uploaded successfully Ready to infer. You selected: {kind, al}'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999, debug=True)
