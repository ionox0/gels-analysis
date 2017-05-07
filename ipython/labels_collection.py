
# coding: utf-8

# In[116]:

import numpy as np
import pandas as pd

pd.options.display.max_columns = 9999
pd.options.display.max_rows = 9999


# In[43]:

labels_full = pd.read_excel('../data/2012-2017_labels.xlsx', sheetname='2016')
labels_full = labels_full.dropna(axis=1, how='all')
labels_full.shape


# In[ ]:

serum_cols = labels_full.columns.str.contains('SPEP')
cols = labels_full.columns[serum_cols]
spreadsheet_top = labels_full.ix[:30, cols]


# In[150]:

labels_left = spreadsheet_top.ix[:,'SPEP':'SPEP.296']


# In[148]:

april_mask = labels_left.ix[0, :]


# In[149]:

april_mask = pd.to_datetime(april_mask)
april_mask = april_mask.isin(pd.date_range('2016-04-01', '2016-04-30'))
april_data = labels_left[april_mask.index[april_mask]]


# In[151]:

# Why is 4-25 2, 3, 0, 4?
# What does '0' signify?
# What does 'o' signify?
april_data


# In[181]:

april_labels = {}
for date in april_data.ix[0,:]:
    mask = april_data.ix[0] == date
    column = april_data.ix[:, mask.index[mask]]
    dz = column.dropna()
    vals = dz.index.values.tolist()
    vals.remove(u'DATE')
    date_str = date.strftime('%Y-%m-%d')
    april_labels[date_str] = vals
    
april_labels


# In[ ]:



