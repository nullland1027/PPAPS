import os
from redis import Redis
from algorithm import utils
from flask import Flask, request, render_template

app = Flask(__name__)  # 申明app对象
redis = Redis(host=os.environ['REDIS_HOST'], port=6379)  # host:数据库的服务名


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/redis')
def hello():
    redis.incr('hits')
    count = redis.get('hits').decode('utf-8')
    return f'Hello! I have been seen {count} times.'


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return 'No file uploaded.', 400

    # 处理上传的文件
    utils.file_process(file)

    return 'File uploaded successfully!'

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8888, debug=True)
