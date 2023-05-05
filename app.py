import os
from tools import Tools
from redis import Redis
from views import bp_views
from model import bp_model
from dataclasses import dataclass
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, send_file, session, g

app = Flask(__name__)  # 申明app对象
app.config['SECRET_KEY'] = "$#%^&YGHG^&(*)IVBIUG*(&RT&(T("
redis = Redis(host='127.0.0.1', port=6379, db=0)  # host:数据库的服务名

# Link other blueprints
app.register_blueprint(bp_views)  # about page render
app.register_blueprint(bp_model)  # about model


@dataclass
class User:
    id: int
    username: str
    password: str


@app.before_request
def before_rqt():
    g.user = None
    if 'user_id' in session:
        pass


@app.route('/login', methods=['POST', 'GET'])
def login():
    """
    处理登陆的函数
    如果已登陆：直接跳转到主页
    """

    if session.get():
        return redirect(url_for('home'))
    if request.method == 'POST':  # received user's input
        user_id = request.form['email']
        pwd = request.form['password']
        if redis.exists(user_id):  # User exists
            if Tools.password_encode(redis.get(user_id)) == Tools.password_encode(pwd):  # True password
                session['user_id'] = user_id  # TODO
                return redirect(url_for("home"))  # Jump to home page
            else:
                pass  # Wrong password
                return redirect(url_for("home_with_error"))
        else:  # User not exists
            pass

    else:  # not receive user's input 由其他页面跳转而来
        return render_template('login.html')


@app.route('/home-page', methods=['POST', 'GET'])
def home():
    """The home page, do prediction"""

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
