from flask import Blueprint
from flask import Flask, request, render_template

bp_views = Blueprint('views', __name__)


@bp_views.route('/')
def index():
    """The website cover page"""
    return render_template('index.html')


@bp_views.route('/signup', methods=['POST', 'GET'])
def sign_up():
    """Sign up page"""
    return render_template('sign-up.html')


@bp_views.route('/after_upload')
def download_page():
    return render_template('download.html')
