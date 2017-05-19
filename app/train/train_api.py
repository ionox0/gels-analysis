import os
import json
from flask import Blueprint, render_template, request

from train_parser import fit_and_save_model
from training_data_uploader import create_train_images



train_api = Blueprint('train_api', __name__)


@train_api.route('/train_images')
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


@train_api.route('/fit_model', methods=['POST'])
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


@train_api.route('/delete_train_images', methods=['DELETE'])
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


@train_api.route('/upload_train_data', methods=['GET', 'POST'])
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


