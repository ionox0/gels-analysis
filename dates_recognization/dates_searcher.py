import re
from datetime import datetime


date_regex = r'^([01]?[0-9])1?([0-2]?[0-9]|3[01])1?([0-9]{2}|19[0-9]{2}|20[0-9]{2})$'


def find_dates(all_date_possibs):
    # Filter matches to ones with b/w 5 and 10 characters
    all_d = filter(lambda x: len(x) > 4 and len(x) < 11, all_date_possibs)

    # Concatenate digit lists into strings
    all_d_cat = [''.join([str(v) for v in d]) for d in all_d]

    # Filter out matches with date_regex
    all_d_filt = filter(lambda x: bool(re.match(date_regex, x)), all_d_cat)

    # Extract date groups
    all_d_match = map(lambda x: re.search(date_regex, x).groups(), all_d_filt)

    # Add first two year digits
    yearfix = [(d[0], d[1], '19' + d[2]) if int(d[2]) > 40 and len(d[2]) == 2 else d for d in all_d_match]
    yearfix = [(d[0], d[1], '20' + d[2]) if int(d[2]) < 40 and len(d[2]) == 2 else d for d in yearfix]

    # Date objs
    try:
        dates = [datetime(int(d[2]), int(d[0]), int(d[1])) if d else None for d in yearfix]
    except:
        return []

    # Formatted
    dates = [d.strftime('%Y-%m-%d') if d else None for d in dates]

    return dates
