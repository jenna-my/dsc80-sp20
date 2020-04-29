
import numpy as np
import os

def data2array(filepath):
    """
    data2array takes in the filepath of a
    data file like `restaurant.csv` in
    data directory, and returns a 1d array
    of data.

    :Example:
    >>> fp = os.path.join('data', 'restaurant.csv')
    >>> arr = data2array(fp)
    >>> isinstance(arr, np.ndarray)
    True
    >>> arr.dtype == np.dtype('float64')
    True
    >>> arr.shape[0]
    100000
    """
    file = open(filepath, 'r')
    skip_bill = file.readline() #skip over column name
    lines = file.readlines()

    lst = []
    #iterate through the lines and append to list
    for line in lines:
        line = line.strip() #get rid of the \n
        value = float(line) #get the float value
        lst.append(value)

    arr = np.asarray(lst)
    return arr

def ends_in_9(arr):
    """
    ends_in_9 takes in an array of dollar amounts
    and returns the proprtion of values that end
    in 9 in the hundredths place.

    :Example:
    >>> arr = np.array([23.04, 45.00, 0.50, 0.09])
    >>> out = ends_in_9(arr)
    >>> 0 <= out <= 1
    True
    """
    multiplier = 100
    check = 10
    end = 9

    multiplied = list(map(lambda x: float(format(x*multiplier, '.2f')) % check, arr))
    check_end = list(filter(lambda x: x == end, multiplied))
    num_ends = len(check_end)

    return num_ends/np.size(arr)
