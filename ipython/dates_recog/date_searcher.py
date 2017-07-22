
# coding: utf-8

# ### Search for date

# In[1]:

import re
from datetime import datetime

date_regex = r'([01]?[0-9])1?([0-3]?[0-9])1?([0-9]{2,4})'


def find_dates(date_possibs):
    # Filter matches to ones with b/w 6 and 8 characters
    all_d = [filter(lambda x: len(x) > 4 and len(x) < 11, date_possibs) for date_possibs in all_date_possibs]

    # Concatenate digit lists into strings
    all_d_cat = [''.join([str(v) for v in x]) for d in all_d for x in d]

    # Filter out matches with date_regex
    all_d_filt = [filter(lambda x: re.match(date_regex, x), d) for d in all_d_cat]

    # Extract date groups
    all_d_match = [map(lambda x: re.search(date_regex, x).groups(), d) for d in all_d_filt]

    # Take 0th elem
    the_d = [d_match[0] if len(d_match) else None for d_match in all_d_match]

    # Date objs
    dates = [datetime(int('20' + d[2]), int(d[0]), int(d[1])) if d else None for d in the_d]

    # Formatted
    dates = [d.strftime('%Y-%m-%d') if d else None for d in dates]

    return dates


# In[ ]:



