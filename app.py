import os
import time
from tools import Tools
from redis import Redis
from views import bp_views
from model import bp_model
from dataclasses import dataclass
from flask import Flask, request, render_template, g, redirect, url_for, send_file, session
from flask import jsonify

app = Flask(__name__)  # 申明app对象
app.config['SECRET_KEY'] = "$#%^&YGHG^&(*)IVBIUG*(&RT&(T("
# redis = Redis(host='172.19.0.2', port=6379, db=0)  # host:数据库的服务名
redis = Redis(host='127.0.0.1', port=6379, db=0)

# Link other blueprints
app.register_blueprint(bp_views)  # about page render
app.register_blueprint(bp_model)  # about model


@dataclass
class User:
    username: str
    password: str


@app.route('/afterLogin', methods=['POST', 'GET'])
def after_login():
    """
    处理登陆的函数
    如果已登陆：直接跳转到主页
    """
    if request.method == 'POST':  # received user's input
        username = request.form['email']
        pwd = request.form['password']
        remember = request.form.get('remember')

        if redis.exists(username):  # User exists
            if redis.get(username).decode('utf-8') == Tools.password_encode(pwd):  # True password
                session['LOGIN_USER'] = username
                return render_template('home-page.html', msg=username)
            else:
                return render_template('login.html', msg='Wrong Password')
        else:  # User not exists
            return render_template('login.html', msg='User does not exist!')
    else:  # not receive user's input 由其他页面跳转而来
        return render_template('login.html')


@app.route('/afterSignUp', methods=['POST'])
def after_sign_up():
    username = request.form['email']
    pwd = request.form['password']
    if redis.exists(username):
        response_data = {"state": '0', "message": "User already exists"}
        return jsonify(response_data), 400
    else:
        redis.set(username, Tools.password_encode(pwd))
        response_data = {"state": '0', "message": "Sign up success"}
        return jsonify(response_data), 200
        # return render_template('login.html')


@app.route('/afterSubmitJob', methods=['POST', 'GET'])
def after_submit_job():
    """The home page, do prediction"""
    kind = request.form.get('kind')
    al = request.form.get('al')
    file = request.files.get('file')

    if not file:
        return render_template('home-page.html')

    file.save(os.path.join("upload", file.filename))
    return redirect(url_for('model.compute', kind=kind, al=al, file_name=file.filename))


@app.route('/check-status')
def check_status():
    if 'LOGIN_USER' not in session.keys():  # No on login
        return render_template('login.html')
    else:
        return render_template('home-page.html', msg=session['LOGIN_USER'])


@app.route('/afterSignOut')
def after_sign_out():
    if 'LOGIN_USER' in session.keys():
        session.pop('LOGIN_USER')
    return render_template('index.html')


@app.route('/showChart')
def show_chart():
    zero = int(redis.get('0'))
    one = int(redis.get('1'))
    return render_template('home-page.html', zero=zero, one=one, click='YES', msg=session['LOGIN_USER'])


@app.route('/download')
def download():
    """The file has been uploaded"""
    file_name = request.args.get('file_name')
    print(file_name)
    return send_file("downloads/" + file_name, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999, debug=True)
