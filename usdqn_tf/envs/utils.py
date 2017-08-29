import numpy as np

def digitize_indexes(labels, bins):
    digit = np.digitize(labels, bins).reshape(-1)
    sort_idx = np.argsort(digit)
    a_sorted = digit[sort_idx]
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    unq_items = a_sorted[unq_first]
    unq_count = np.diff(np.nonzero(unq_first)[0])
    return np.split(sort_idx, np.cumsum(unq_count))
