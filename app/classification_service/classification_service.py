import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

from app.labels_collector.labels_collector import get_labels, get_dz_labels

from matplotlib import pyplot as plt


### Classification
def auto_classify(X_flat, y):
    x_train, x_test, y_train, y_test = train_test_split(X_flat, y)

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print score

    plt.figure(figsize=(20, 16))

    X_means_df = pd.DataFrame(X_flat)

    ctrl = X_means_df[np.array(y) == 0]
    print len(ctrl), len(y)
    plt.subplot(211)
    ctrl.T.plot(alpha=.1, color='blue', ax=plt.gca(), legend=None, label='ctrl')

    dz = X_means_df[np.array(y) == 1]
    print len(dz), len(y)
    plt.subplot(212)
    dz.T.plot(alpha=.1, color='red', ax=plt.gca(), legend=None, label='dz')

    plt.show()
