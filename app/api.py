import os
import json
import StringIO
# from PIL import Image
from flask import Flask
from flask import request
from flask import send_file
from nocache import nocache

import sys
sys.path.insert(0, '')
from classifier import manual_classify_gel, auto_classify_gel
from manual_train.train_api import train_api



app = Flask(__name__, static_url_path='', static_folder='.')
app.register_blueprint(train_api)


@app.route('/')
def root():
    return app.send_static_file('test.html')


@app.route('/result')
@nocache
def result():
    return app.send_static_file('labeled_image.jpg')


@app.route('/auto_classify', methods=['POST'])
def auto_classify_route():
    """
    Flask route for automatic abnormal band detection
    :return: Labeled image
    """
    if 'file' not in request.files:
        raise Exception('no file provided')

    file = request.files['file']
    if file:
        # filename = secure_filename(file.filename)
        src = os.getcwd() + '/uploaded_data/'
        file.save(os.path.join(src, file.filename))

    rois = json.loads(request.form['rois'])
    preds = auto_classify_gel(file.filename, rois)
    preds = json.dumps(list(preds))

    return preds


@app.route('/manual_classify', methods=['GET', 'POST'])
def manual_classify_route():
    """
    Flask route for manual abnormal band detection
    :return: Labeled image
    """
    if 'file' not in request.files:
        raise Exception('no file provided')

    file = request.files['file']
    if file:
        # filename = secure_filename(file.filename)
        src = os.getcwd() + '/uploaded_data/'
        file.save(os.path.join(src, file.filename))

    rois = json.loads(request.form['rois'])
    min_y_value = int(request.form['min-y-value'])
    max_y_value = int(request.form['max-y-value'])
    threshold = float(request.form['threshold'])

    danger_zone = {
        'y_min': int(min_y_value),
        'y_max': int(max_y_value),
    }

    result_filename = manual_classify_gel(file.filename, rois, danger_zone, threshold)

    return send_file(result_filename, mimetype='image/gif')


def serve_pil_image(pil_img):
    img_io = StringIO.StringIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


def strtobool(val):
    """
    https://github.com/python-git/python/blob/master/Lib/distutils/util.py
    Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = str.lower(str(val))
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise (ValueError, "invalid truth value %r" % (val,))


if __name__ == '__main__':
    env_port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=env_port, debug=True)
