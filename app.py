from flask import Flask
from redis import Redis


app = Flask(__name__)  # 申明app对象
redis = Redis(host='my_redis', port=6379)


@app.route('/')
def hello():
    redis.incr('hits')
    return f"Hello Container World! 点击次数: {redis.get('hits').decode('utf-8')} 次\n"

