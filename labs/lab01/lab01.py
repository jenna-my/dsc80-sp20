
import os

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question # 0
# ---------------------------------------------------------------------

def consecutive_ints(ints):
    """
    consecutive_ints tests whether a list contains two
    adjacent elements that are consecutive integers.

    :param ints: a list of integers
    :returns: a boolean value if ints contains two
    adjacent elements that are consecutive integers.

    :Example:
    >>> consecutive_ints([5,3,6,4,9,8])
    True
    >>> consecutive_ints([1,3,5,7,9])
    False
    """

    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def median(nums):
    """
    median takes a non-empty list of numbers,
    returning the median element of the list.
    If the list has even length, it should return
    the mean of the two elements in the middle.

    :param nums: a non-empty list of numbers.
    :returns: the median of the list.

    :Example:
    >>> median([6, 5, 4, 3, 2]) == 4
    True
    >>> median([50, 20, 15, 40]) == 30
    True
    >>> median([1, 2, 3, 4]) == 2.5
    True
    """
    #sort in ascending order
    nums.sort()

    if len(nums) % 2 == 0: #if list has even length
        elem1 = nums[(len(nums) // 2) - 1]
        elem2 = nums[len(nums) // 2]
        return (elem1 + elem2) / 2
    else:
        return nums[len(nums) // 2]

# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def same_diff_ints(ints):
    """
    same_diff_ints tests whether a list contains
    two list elements i places apart, whose distance
    as integers is also i.

    :param ints: a list of integers
    :returns: a boolean value if ints contains two
    elements as described above.

    :Example:
    >>> same_diff_ints([5,3,1,5,9,8])
    True
    >>> same_diff_ints([1,3,5,7,9])
    False
    """
    if len(ints) == 0:
        return False
    for i in range(1, len(ints)): #represent num of places apart
        for j in range(len(ints)): #iterate through each element in list
            if j + i >= len(ints): #no more elems to look at for this iteration
                break
            elif abs(ints[j] - ints[j + i]) == i: #equal places and distances
                return True
    return False

# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def prefixes(s):
    """
    prefixes returns a string of every
    consecutive prefix of the input string.

    :param s: a string.
    :returns: a string of every consecutive prefix of s.

    :Example:
    >>> prefixes('Data!')
    'DDaDatDataData!'
    >>> prefixes('Marina')
    'MMaMarMariMarinMarina'
    >>> prefixes('aaron')
    'aaaaaraaroaaron'
    """
    output = ''
    for i in range(len(s) + 1):
        add = s[0:i]
        output += add
    return output

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def evens_reversed(N):
    """
    evens_reversed returns a string containing
    all even integers from  1  to  N  (inclusive)
    in reversed order, separated by spaces.
    Each integer is zero padded.

    :param N: a non-negative integer.
    :returns: a string containing all even integers
    from 1 to N reversed, formatted as decsribed above.

    :Example:
    >>> evens_reversed(7)
    '6 4 2'
    >>> evens_reversed(10)
    '10 08 06 04 02'
    """

    if N == 0 or N == 1: #nothing returned
        return ''

    output = ''
    num_pads = len(str(N)) #num of digits that will be length of each integer

    for i in range(N, 1, -1):
        if (i % 2) == 0:
            str_int = str(i)
            str_int = str_int.zfill(num_pads) #zeropad the integer
            output += str_int
            output += ' '

    output = output.strip() #remove whitespace
    return output


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def last_chars(fh):
    """
    last_chars takes a file object and returns a
    string consisting of the last character of the line.

    :param fh: a file object to read from.
    :returns: a string of last characters from fh

    :Example:
    >>> fp = os.path.join('data', 'chars.txt')
    >>> last_chars(open(fp))
    'hrg'
    """
    output = ''
    with open(fh.name) as fh:
        for line in fh:
            output += line[-2]
    return output


# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------

def arr_1(A):
    """
    arr_1 takes in a numpy array and
    adds to each element the square-root of
    the index of each element.

    :param A: a 1d numpy array.
    :returns: a 1d numpy array.

    :Example:
    >>> A = np.array([2, 4, 6, 7])
    >>> out = arr_1(A)
    >>> isinstance(out, np.ndarray)
    True
    >>> np.all(out >= A)
    True
    """
    copy = np.copy(A)
    indices = np.array(range(copy.size)) #array of indices
    roots = np.sqrt(indices)
    return copy + roots

def arr_2(A):
    """
    arr_2 takes in a numpy array of integers
    and returns a boolean array (i.e. an array of booleans)
    whose ith element is True if and only if the ith element
    of the input array is divisble by 16.

    :param A: a 1d numpy array.
    :returns: a 1d numpy boolean array.

    :Example:
    >>> out = arr_2(np.array([1, 2, 16, 17, 32, 33]))
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('bool')
    True
    """
    B = (A % 16 == 0)
    return B

def arr_3(A):
    """
    arr_3 takes in a numpy array of stock
    prices per share on successive days in
    USD and returns an array of growth rates.

    :param A: a 1d numpy array.
    :returns: a 1d numpy array.

    :Example:
    >>> fp = os.path.join('data', 'stocks.csv')
    >>> stocks = np.array([float(x) for x in open(fp)])
    >>> out = arr_3(stocks)
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('float')
    True
    >>> out.max() == 0.03
    True
    """
    copy = np.copy(A)

    diff = np.ediff1d(copy) #differences
    copy = np.delete(copy, copy.size - 1) #remove last elem
    diff = diff/copy
    output = np.round(diff, 2)
    return output


def arr_4(A):
    """
    Create a function arr_4 that takes in A and
    returns the day on which you can buy at least
    one share from 'left-over' money. If this never
    happens, return -1. The first stock purchase occurs on day 0
    :param A: a 1d numpy array of stock prices.
    :returns: an integer of the total number of shares.

    :Example:
    >>> import numbers
    >>> stocks = np.array([3, 3, 3, 3])
    >>> out = arr_4(stocks)
    >>> isinstance(out, numbers.Integral)
    True
    >>> out == 1
    True
    """
    copy = np.copy(A)
    leftovers = 20 % A
    total_leftovers = np.cumsum(leftovers) #total leftovers after each day
    enough_left = total_leftovers >= A #boolean array to determine whether we have enough to buy another stock for the day

    if True in enough_left:
        day = list(enough_left).index(True) #gets first index where we can buy
        return day
    else:
        return -1


# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def movie_stats(movies):
    """
    movies_stats returns a series as specified in the notebook.

    :param movies: a dataframe of summaries of
    movies per year as found in `movies_by_year.csv`
    :return: a series with index specified in the notebook.

    :Example:
    >>> movie_fp = os.path.join('data', 'movies_by_year.csv')
    >>> movies = pd.read_csv(movie_fp)
    >>> out = movie_stats(movies)
    >>> isinstance(out, pd.Series)
    True
    >>> 'num_years' in out.index
    True
    >>> isinstance(out.loc['second_lowest'], str)
    True
    """
    def num_years():
        """number of years covered in the dataset"""
        years = movies['Year']
        return ('num_years', years.nunique())

    def tot_movies():
        """total number of movies over all years"""
        total = movies['Number of Movies'].sum()
        return ('tot_movies', total)

    def yr_fewest_movies():
        """year with fewest movies made (earliest)"""
        copy = movies.copy()
        year = copy.sort_values(['Number of Movies', 'Year']).reset_index(drop = True).Year.loc[0]
        return ('yr_fewest_movies', year)

    def avg_gross():
        """average amount of money grossed over all years"""
        avg = movies['Total Gross'].mean()
        if avg is np.nan:
            raise
        return ('avg_gross', avg)

    def highest_per_movie():
        """The year with the highest gross per movie"""
        copy = movies.copy()
        copy['Gross Per Movie'] = copy['Total Gross'] / copy['Number of Movies'] #calculate gross per movie
        year = copy.sort_values(['Gross Per Movie'], ascending = False).reset_index(drop = True).Year.loc[0]
        return ('highest_per_movie', year)

    def second_lowest():
        """name of the top movie during the second lowest (total) grossing year"""
        copy = movies.copy()
        name = copy.sort_values(['Total Gross']).reset_index(drop = True)['#1 Movie'].loc[1]
        return ('second_lowest', name)

    def avg_after_harry():
        """avg number of movies made the year after an HP movie was #1 movie"""
        copy = movies.copy()
        copy = copy.sort_values(['Year']).reset_index(drop = True) #years early to present
        harry_years = copy[copy['#1 Movie'].str.contains('Harry')].Year #years where harry potter was #1
        next_years = harry_years + 1
        check = list(next_years.values)
        next_years_df = copy[copy['Year'].isin(check)]
        avg = next_years_df['Number of Movies'].mean()
        if avg is np.nan:
            raise
        return ('avg_after_harry', avg)

    functions = [num_years(), tot_movies(), yr_fewest_movies(), avg_gross(), highest_per_movie(), second_lowest(), avg_after_harry()]
    stats = {}
    for fxn in functions:
        try:
            info = fxn
            stats[info[0]] = info[1]
        except:
            continue
    series = pd.Series(stats)

    return series

# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def parse_malformed(fp):
    """
    Parses and loads the malformed csv data into a
    properly formatted dataframe (as described in
    the question).

    :param fh: file handle for the malformed csv-file.
    :returns: a Pandas DataFrame of the data,
    as specificed in the question statement.

    :Example:
    >>> fp = os.path.join('data', 'malformed.csv')
    >>> df = parse_malformed(fp)
    >>> cols = ['first', 'last', 'weight', 'height', 'geo']
    >>> list(df.columns) == cols
    True
    >>> df['last'].dtype == np.dtype('O')
    True
    >>> df['height'].dtype == np.dtype('float64')
    True
    >>> df['geo'].str.contains(',').all()
    True
    >>> len(df) == 100
    True
    >>> dg = pd.read_csv(fp, nrows=4, skiprows=10, names=cols)
    >>> dg.index = range(9, 13)
    >>> (dg == df.iloc[9:13]).all().all()
    True
    """
    row_data=[]
    cols = ['first', 'last', 'weight', 'height', 'geo']

    with open(fp) as fh:

        fh.readline() #skip first line (column names)

        for line in fh:
            #remove newline character
            line = line.strip('\n')
            elems = line.split(",") #all entries converted to string - last col value not split
            elems = list(filter(None, elems)) #remove empty entries

            #there should be 6 entries in the list now
            for i in range(6):
                elems[i] = elems[i].replace('"', '') #drop quotations in all values
                if i == 2 or i == 3: #float columns
                    elems[i] = float(elems[i])

            #combine index 4 and 5 to make geo column
            elems[4] = elems[4] + ',' + elems[5]
            del elems[-1]
            #add to row data
            row_data.append(elems)

    #make new dataframe
    output = pd.DataFrame(row_data, columns = cols)
    return output



# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q00': ['consecutive_ints'],
    'q01': ['median'],
    'q02': ['same_diff_ints'],
    'q03': ['prefixes'],
    'q04': ['evens_reversed'],
    'q05': ['last_chars'],
    'q06': ['arr_%d' % d for d in range(1, 5)],
    'q07': ['movie_stats'],
    'q08': ['parse_malformed']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """

    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" % (q, elt)
                raise Exception(stmt)

    return True
