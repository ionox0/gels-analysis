import os
import pickle
import numpy as np
import pandas as pd
from skimage import data
from skimage.transform import resize
from skimage.filters import threshold_otsu, threshold_adaptive

# Features
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import f_classif, RFE, SelectFdr, SelectFpr, SelectFromModel, SelectKBest, SelectPercentile
from sklearn.preprocessing import RobustScaler, StandardScaler, Imputer, MaxAbsScaler, MinMaxScaler

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectPercentile, VarianceThreshold

# Models - Linear
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LassoCV, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# Models - Non-Linear
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV

# Testing
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

from sklearn.externals import joblib


train_image_files = [x for x in os.listdir('../app/train_images/') if '.jpg' in x]

input_shape = (233, 50, 4)
input_dim = (50, 233)
num_classes = 2


def get_training_data():
    train_images = []
    train_labels = []
    for f in train_image_files:
        img = data.imread('../app/train_images/' + f)
        train_images.append(img)
        if 'DZ' in f:
            train_labels.append(1)
        else:
            train_labels.append(0)
    return np.array(train_images), np.array(train_labels)


def do_threshold(img):
    thresh = threshold_otsu(img)
#     thresh = threshold_adaptive(image, 15, 'mean')
    binary = img > thresh
    return np.array(binary)


def collapse_whitespace_margins(img, img_thresholded):
    vert_sum_img = img_thresholded.sum(axis=0)
    img_rolled = np.rollaxis(vert_sum_img, -1)
    max_intensity = np.max(vert_sum_img)
    margin_indices = np.where(img_rolled[2] == max_intensity)[0]

    mask_array = np.ones(img.shape[1], dtype=bool)
    mask_array[margin_indices] = False

    return np.array(img[:, mask_array, :])


def collapse_bottom_margins(img, img_thresholded):
    horiz_sum_img = img_thresholded.sum(axis=1)
    img_rolled = np.rollaxis(horiz_sum_img, -1)
    max_intensity = np.max(horiz_sum_img)

    mask_array = np.ones(img.shape[0], dtype=bool)
    bottom_margin_indices = []

    # todo - correct color channel?
    if img_rolled[0][0] != max_intensity:
        # Albumin reached bottom
        return img
    for i, k in enumerate(np.flip(img_rolled[0], axis=0)):
        if k == max_intensity:
            bottom_margin_indices.append(img_rolled[0].shape[0] - i - 1)
        else:
            break

    mask_array[bottom_margin_indices] = False
    return np.array(img[mask_array, :, :])


# Resize images
def resize_images(imgs):
    imgs_resized = []
    for img in imgs:
        img_resized = resize(img, input_dim)
        imgs_resized.append(img_resized)
    return np.array(imgs_resized)


# Mean across horizontal axis
def calc_lane_means(lanes):
    lanes_means = []
    for lane in lanes:
        lane_mean = lane.mean(axis=1)
        # Just take the blue channels
        lanes_means.append(lane_mean[:, 2])

    return lanes_means


def smooth_lanes(X, N):
    # plt.figure(figsize=(15, 6))
    # plt.plot(X_means[0])

    smootheds = []
    for n in N:
        smoothed = [pd.rolling_mean(x, n)[n - 1:] for x in X]
        smootheds.append(smoothed)
        # plt.plot(smoothed[0])
        # plt.legend()

    # plt.show()
    return smootheds


def grid_search_metrics(pipe, param_grid, x_train, y_train, x_test, y_test):
    grid = GridSearchCV(pipe, param_grid=param_grid, scoring='roc_auc')
    grid.fit(x_train, y_train)
    print("Best GS score: {}".format(grid.best_score_))
    print("Best params: {}".format(grid.best_params_))
    score = grid.score(x_test, y_test)
    print("Best test score: {}".format(score))
    print("Overfitting amount: {}".format(grid.best_score_ - score))
    return grid.best_score_, score


def get_pipe_and_grid():
    pipe = Pipeline([
        ("variance", VarianceThreshold()),
        ("selection_1", SelectKBest(score_func=f_classif)),
        ("polys", PolynomialFeatures()),
        ("scaling", StandardScaler()),
        ("selection_2", SelectKBest(score_func=f_classif)),
        ("model", RandomForestClassifier(class_weight={0: 1, 1: 3}))
    ])

    param_grid = {
        'selection_1__k': [45],  # [45, 50, 55],
        'selection_2__k': [600],  # [600, 700, 800],

        "model__max_depth": [None],  # [3, None],
        "model__max_features": [13],  # [13, 14, 15],
        "model__min_samples_split": [3],  # [2, 3, 4],
        "model__min_samples_leaf": [10],  # [7, 10, 15],
        "model__bootstrap": [False],  # [True, False],
        "model__criterion": ['entropy'],  # ["gini", "entropy"]
    }

    return pipe, param_grid


def do_fit_and_save_model(pipe, X_final, y_final):
    pipe_fitted = pipe.fit(X_final, y_final)
    s = pickle.dumps(pipe_fitted)

    # clf2 = pickle.loads(s)
    # clf2.predict(X[0:1])
    joblib.dump(pipe, 'trained_classifier.pkl')


def fit_and_save_model():
    X, y = get_training_data()
    print X.shape, y.shape

    y_df = pd.DataFrame(y)
    print y_df[0].value_counts()

    X_threshold = [do_threshold(x).astype(np.uint16) for x in X]
    X_threshold = np.array(X_threshold)
    print len(X_threshold)

    X_collapsed = [collapse_whitespace_margins(x, z) for x, z in zip(X, X_threshold)]
    X_collapsed_vert = [collapse_bottom_margins(x, z) for x, z in zip(X_collapsed, X_threshold)]
    X_resized = resize_images(X_collapsed_vert)
    print len(X_resized)

    X_means = np.array(calc_lane_means(X_resized))
    print len(X_means), X_means[0].shape, len(y)

    # X_smoothed = smooth_lanes(X_means, [2, 5, 10])[0]

    # Convert to DF in order to map back to original images
    X_means_df = pd.DataFrame(X_means)

    X_final = X_means_df
    y_final = y_df[0]
    x_train, x_test, y_train, y_test = train_test_split(X_final, y_final, stratify=y_df[0])

    # y_train = to_categorical(y_train, num_classes)
    # y_test = to_categorical(y_test, num_classes)

    print x_train[0].shape, x_test[0].shape, y_train.shape, y_test.shape

    pipe, param_grid = get_pipe_and_grid()
    train_score, test_score = grid_search_metrics(pipe, param_grid, x_train, y_train, x_test, y_test)

    do_fit_and_save_model(pipe, X_final, y_final)
    return train_score, test_score