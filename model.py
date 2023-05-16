import os
from flask import Blueprint
from flask import Flask, request, render_template, redirect, url_for
from algorithm.interface import file_md5, rf_pred, xgb_pred, lgbm_pred, attention_pred

bp_model = Blueprint('model', __name__)


@bp_model.route('/computing')
def compute():
    """Compute the result and generate the target file in `downloads`"""
    kind = request.args.get('kind')
    al = request.args.get('al')
    file_name = request.args.get('file_name')  # [:-4]  # without `.csv`

    if al == 'random_forest':
        new_file_name = rf_pred(os.path.join("upload", file_name), kind)  # the generated file
        if new_file_name != 'FAILED':
            return redirect(url_for('download', file_name=new_file_name))
        else:
            return "File ERROR"
    elif al == 'xgboost':
        new_file_name = xgb_pred(os.path.join("upload", file_name), kind)  # the generated file
        if new_file_name != 'FAILED':
            return redirect(url_for('download', file_name=new_file_name))
        else:
            return "File ERROR"
    elif al == 'lightgbm':
        new_file_name = lgbm_pred(os.path.join("upload", file_name), kind)  # the generated file
        if new_file_name != 'FAILED':
            return redirect(url_for('download', file_name=new_file_name))
        else:
            return "File ERROR"
    elif al == 'attention_network':
        new_file_name = attention_pred(os.path.join("upload", file_name), kind)
        if new_file_name != 'FAILED':
            return redirect(url_for('download', file_name=new_file_name))
        else:
            return "File ERROR"
    else:
        raise AttributeError
