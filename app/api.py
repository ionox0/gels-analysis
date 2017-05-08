import os
import json
import StringIO
# from PIL import Image
from flask import Flask, render_template
from flask import request
from flask import send_file
from nocache import nocache


from analyzer import analyze_gel, auto_classify_gel
from train_parser import fit_and_save_model
from training_data_uploader import create_train_images


app = Flask(__name__, static_url_path='', static_folder='.')


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/train_images')
def dirtree():
    '''
    Route to show the existing training image files
    :return:
    '''
    path = os.path.expanduser(u'./train_images')
    return render_template('dirtree.html', tree=make_tree(path))

def make_tree(path):
    tree = dict(name=os.path.basename(path), children=[])
    try: lst = os.listdir(path)
    except OSError:
        pass #ignore errors
    else:
        for name in lst:
            fn = os.path.join(path, name)
            if os.path.isdir(fn):
                tree['children'].append(make_tree(fn))
            else:
                tree['children'].append(dict(name=name))
    return tree


@app.route('/delete_train_images', methods=['DELETE'])
def delete_train_images_route():
    '''
    Route to show the existing training image files
    :return:
    '''
    folder = './train_images'
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

    return 'success'


@app.route('/result')
@nocache
def result():
    return app.send_static_file('labeled_image.jpg')


@app.route('/upload_train_data', methods=['GET', 'POST'])
def upload_train_data_route():
    """
    Flask route for uploading training data
    :return: success bool
    """
    if 'picture' not in request.files:
        raise Exception('no file provided')

    file = request.files['picture']
    filename = request.form['filename']
    if file:
        # filename = secure_filename(file.filename)
        src = os.getcwd() + '/full_gels/'
        file.save(os.path.join(src, filename))

    rois = json.loads(request.form['rois'])
    labels = json.loads(request.form['labels'])

    result = create_train_images(filename, rois, labels)
    return 'success'


@app.route('/fit_model', methods=['POST'])
def fit_model_route():
    """
    Flask route for fitting the classifier on the training images
    :return:
    """
    train_score, test_score = fit_and_save_model()
    scores = {
        'train': train_score,
        'test': test_score
    }
    return json.dumps(scores)


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

    result_filename = analyze_gel(file.filename, rois, danger_zone, threshold)

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
