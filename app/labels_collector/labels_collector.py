import numpy as np
import pandas as pd
from datetime import datetime


pd.options.display.max_columns = 9999
pd.options.display.max_rows = 9999

# Read Excel File
labels_full = pd.read_excel('../data/2012-2017_labels.xlsx', sheetname='2016')

# Drop Missing all
labels_full = labels_full.dropna(axis=1, how='all')

# Why is 4-25 2, 3, 0, 4?
# What does '0' signify?
# What does 'o' signify?


def get_labels(start_date, end_date):
    if isinstance(start_date, datetime):
        start_date = datetime.strftime(start_date, '%Y-%m-%d')
    if end_date(end_date, datetime):
        end_date = datetime.strftime(end_date, '%Y-%m-%d')

    labels = labels_full.copy()
    # Take SPEP Cols, not IFE
    serum_cols = labels.columns.str.contains('SPEP')
    cols = labels.columns[serum_cols]

    labels = labels.reset_index()
    # Take just PEP #1 for each date (todo - no hard coding plz)
    labels = labels.loc[:35, cols]
    # Use DATE row as column indices
    labels.columns = pd.to_datetime(labels.loc[0, :], errors='coerce')
    # Drop DATE, and Gel # rows (todo - ditto)
    labels.drop(labels.index[:2], inplace=True)

    # shift index to align with first control lane (#2)
    ctrl_lane_idx = labels.iloc[:, 1].first_valid_index()
    shift_val = ctrl_lane_idx - 2
    labels = labels.set_index(labels.index - shift_val)

    labels = labels.loc[:, start_date: end_date]
    return labels


def get_dz_labels(labels):
    dz_labels = {}
    for col in labels.columns:
        date = col.strftime('%Y-%m-%d')
        inds = labels[col].index.values
        dz = inds[labels[col].notnull().tolist()].tolist()
        dz_labels[date] = dz
    return dz_labels


def retreive_labels(start_date, end_date):
    '''
    Load Labels from pre-existing Excel spreadsheet
    :param start_date:
    :param end_date:
    :return:
    '''
    labels = get_labels(start_date, end_date)
    dz_labels = get_dz_labels(labels)

    return dz_labels


# od = collections.OrderedDict(sorted(nov_2016_dz_labels.items()))
# asdf = [1,6,12,21,41,42,51,52,56,83,84,89,90,96,97,106,123,131,136,152,153,156,157] # 7, 22
#
# blah = zip([str(i + 1) + ' ' + str(val) for i, val in enumerate(asdf)], od.items())
#
# print blah


def build_labels(labels, date, lanes):
    '''
    Turn values from Excel spreadsheet into 0 (ctrl) or 1 (dz), and return in list
    :return:
    '''
    y = []
    for i in range(len(lanes)):
        if i in labels[date]:
            y.append(1)
        else:
            y.append(0)

    return y


