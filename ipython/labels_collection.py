
# coding: utf-8

# In[211]:

import numpy as np
import pandas as pd
from datetime import datetime

pd.options.display.max_columns = 9999
pd.options.display.max_rows = 9999


# In[212]:

# Read Excel File
labels_full = pd.read_excel('../data/2012-2017_labels.xlsx', sheetname='2016')

# Drop Missing all
labels_full = labels_full.dropna(axis=1, how='all')

# Why is 4-25 2, 3, 0, 4?
# What does '0' signify?
# What does 'o' signify?


# In[221]:

labels_full.ix[:, 'SPEP.260']


# In[222]:

def get_labels(start_date, end_date):
    labels = labels_full.copy()
    # Take SPEP Cols, not IFE
    serum_cols = labels.columns.str.contains('SPEP')
    cols = labels.columns[serum_cols]

    labels = labels.reset_index()
    # Take just PEP #1 for each date (todo - no hard coding plz)
    labels = labels.loc[:35, cols]
    # Use DATE row as column indices
    labels.columns = pd.to_datetime(labels.loc[0,:], errors='coerce')
    # Drop DATE, and Gel # rows (todo - ditto)
    labels.drop(labels.index[:2], inplace=True)
    
    # shift index to align with first control lane (#2)
    ctrl_lane_idx = labels.iloc[:,1].first_valid_index()
    shift_val = ctrl_lane_idx - 2
    labels = labels.set_index(labels.index - shift_val)
    
    labels = labels.loc[:,start_date : end_date]
    return labels


# In[214]:

def get_dz_labels(labels):
    dz_labels = {}
    for col in labels.columns:
        date = col.strftime('%Y-%m-%d')
        inds = labels[col].index.values
        dz = inds[labels[col].notnull().tolist()].tolist()
        dz_labels[date] = dz
    return dz_labels


# In[223]:

april_2016_labels = get_labels(datetime(2016, 4, 1), datetime(2016, 4, 30))
april_2016_dz_labels = get_dz_labels(april_2016_labels)

nov_2016_labels = get_labels(datetime(2016, 11, 1), datetime(2016, 11, 30))
nov_2016_dz_labels = get_dz_labels(nov_2016_labels)


# In[224]:

nov_2016_labels


# In[ ]:



