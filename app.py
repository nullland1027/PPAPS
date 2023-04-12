import os
import pickle
import pandas as pd
from redis import Redis
from views import bp_views
from model import bp_model
from algorithm.utilitis import file_md5, rf_pred
from algorithm.ml_preds import RFPredictor, LGBMPredictor
from flask import Flask, request, render_template, send_from_directory
from flask import redirect, url_for, send_file

app = Flask(__name__)  # 申明app对象
redis = Redis(host='127.0.0.1', port=6379)  # host:数据库的服务名

# Global variables area
IS_LOGIN = False  # 用户是否登陆

# Link other blueprints
app.register_blueprint(bp_views)  # about page render
app.register_blueprint(bp_model)  # about model


@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


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


@app.route('/home-page', methods=['POST', 'GET'])
def home():
    """The home page, do prediction"""
    global IS_LOGIN
    if not IS_LOGIN:  # 没有登陆
        return redirect(url_for("login"))

    kind = request.form.get('kind')
    al = request.form.get('al')
    file = request.files.get('file')

    if not file:
        return render_template('home-page.html')

    file.save(os.path.join("upload", file.filename))
    print('in home 函数', file.filename)
    return redirect(url_for('model.compute', kind=kind, al=al, file_name=file.filename))


@app.route('/download')
def download():
    """The file has been uploaded"""
    file_name = request.args.get('file_name')
    return send_file("downloads/" + file_name, as_attachment=True)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999, debug=True)
