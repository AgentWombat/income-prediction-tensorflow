import pandas as pd
import numpy as np

def load_data():
    '''
    Gets income data.
    :returns: A touple of touples--(x_train, y_train), (x_test, y_test)--which
        contain the training and test data respectively. The data are in the
        form of Numpy arrays. The input data are arranged in vectors like so:
        [age, college_degree, currently_married, sex (1 = male), hours_per_week_work]
        The output vectors like so: [makes_more_than_50k]
    '''

    df = pd.read_csv('adult.csv')

    df['college.degree'] = list(map(lambda x: x >= 11, df['education.num']))

    df['currently.married'] = list(map(lambda x: 1 if "married" in x.lower() else 0,
        df['marital.status']))

    df['sex.num'] = list(map(lambda x: 1 if x == 'Male' else 0, df['sex']))
        
    df['greater.than.50k'] = list(map(lambda x: 1 if ">" in x else 0, df['income']))

    x = np.array(list(zip(df['age'],df['college.degree'],df['currently.married'],df['sex.num'],
        df['hours.per.week'])), dtype = 'float32')

    y = np.array(df['greater.than.50k'], dtype = 'float32')

    # Ensure equal number of positive and negative examples
    x_stripped = []
    y_stripped = []

    pos = sum(y)
    neg = 0

    for i, val in enumerate(y):

        if val == 1:
            y_stripped.append(val)
            x_stripped.append(x[i])

        elif neg < pos:
            y_stripped.append(val)
            x_stripped.append(x[i])
            neg += 1

    seed = np.random.randint(935115)

    # Randomize ordering of examples
    x_stripped = np.random.RandomState(seed=seed).permutation(x_stripped)
    y_stripped = np.random.RandomState(seed=seed).permutation(y_stripped)

    return x_stripped[:-300], y_stripped[:-300], x_stripped[-300:], y_stripped[-300:]


# This is for standardizing data
MAX_AGE = 90
MAX_HOURS = 60
