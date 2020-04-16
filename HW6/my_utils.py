import os
import pickle
from collections import deque
from functools import wraps

import matplotlib
import numpy as np
import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import FunctionTransformer

from consts import TEXT_COLUMN


def save(object_to_save, name: str) -> None:
    """Save object `object_to_save` to pickle file with name data/{name}.pickle"""
    with open('data/{}.pickle'.format(name), 'wb') as handle:
        pickle.dump(object_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(name: str):
    """Load object from pickle file with name data/{name}.pickle"""
    with open('data/{}.pickle'.format(name), 'rb') as handle:
        return pickle.load(handle)


def save_result(file_name: str = None, calculate: bool = False, skip: bool = False):
    """
    Decorator for saving/load results of wrapped function to the file with `filename`.ext

    if calculate == true wrapped function will be run and result will be saved
    else function call will be skipped and result will be loaded if file is exists

    if skip == true function call skipped as well as file loading

    NB!
    path for save:
        1. data/{file_name}.parquet if type of result is pd.DataFrame
        2. img/{file_name}.png and data/{file_name}.pickle if type of result is matplotlib.figure.Figure
        3. Else data/{file_name}.pickle

    NB!
    the final file path can be overwrited by kwargs['file_name'] if it exists
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            final_file_name = kwargs['file_name'] if 'file_name' in kwargs else file_name

            if calculate:
                result = func(*args, **kwargs)
                if final_file_name:
                    if isinstance(result, pd.DataFrame):
                        result.to_parquet('data/{}.parquet'.format(final_file_name))
                    elif isinstance(result, matplotlib.figure.Figure):
                        result.savefig('img/{}.png'.format(result._suptitle._text), dpi=500)
                        save(result, result._suptitle._text)
                        return
                    else:
                        save(result, final_file_name)
                return result
            else:
                if skip:
                    print('code skipped')
                    return

                if 'title' in kwargs:
                    print('read {}'.format(kwargs['title']))
                    return load(kwargs['title'])

                if final_file_name:
                    print('read {}'.format(final_file_name))
                    if os.path.isfile('data/{}.parquet'.format(final_file_name)):
                        return pd.read_parquet('data/{}.parquet'.format(final_file_name))
                    else:
                        return load(final_file_name)

        return wrapper

    return decorator


@save_result('mappings')
def generate_mappings(my_data: pd.DataFrame):
    """Generate label mappings according to hierarchies"""

    def generate_new_clusters(obj, base_mapping):
        for v in base_mapping.values():
            for i in obj:
                if v.startswith(i):
                    yield i

    a = [np.unique(['.'.join(i.split('.')[:k]) for i in my_data.target_names]) for k in range(1, 4)]
    targets = np.unique(my_data.target)
    mapping1 = dict(zip(targets, a[2]))
    mapping2 = dict(zip(targets, generate_new_clusters(a[1], mapping1)))
    mapping3 = dict(zip(targets, generate_new_clusters(a[0], mapping1)))
    return mapping1, mapping2, mapping3


def word_lemmatizer(word):
    wordnet_lemmatizer = WordNetLemmatizer()
    word = word.replace('_', '')
    word1 = wordnet_lemmatizer.lemmatize(word, pos="n")  # NOUNS
    word2 = wordnet_lemmatizer.lemmatize(word1, pos="v")  # VERB
    return wordnet_lemmatizer.lemmatize(word2, pos=("a"))  # ADJ


def text_lemmatizer(text):
    return ' '.join(map(word_lemmatizer, text))


def lemmatizer(x):
    x[TEXT_COLUMN] = x[TEXT_COLUMN].apply(lambda text: text_lemmatizer(tokenize(remove_stopwords(text))))
    return x


def get_sentimnent(x):
    vader_analyzer = SentimentIntensityAnalyzer()
    x['sentimnent'] = x[TEXT_COLUMN].apply(lambda text: vader_analyzer.polarity_scores(text))
    return x


def text_feature_selector(x):
    x['word_count'] = x[TEXT_COLUMN].apply(lambda text: len(str(text).split()))
    x['length'] = x[TEXT_COLUMN].apply(len)
    x['word_density'] = x['length'] / x['word_count']
    return x


# In[17]:


@save_result('preprocessed_df', calculate=False)
def nlp_preprocessing(df):
    print(df.shape)
    df = FunctionTransformer(lemmatizer, validate=False).transform(df)
    print(df.shape)
    df = FunctionTransformer(text_feature_selector, validate=False).transform(df)
    print(df.shape)
    df = FunctionTransformer(get_sentimnent, validate=False).transform(df)
    print(df.shape)
    df = pd.concat([df, pd.io.json.json_normalize(df['sentimnent'])], axis=1).drop(columns=['sentimnent'])
    print(df.shape)
    return df


def dbscan_grid_search(X_data, lst, eps_space=(0.5, ),
                       min_samples_space=(5, ), min_clust=7, max_clust=20, noise_b=0.25):
    """
    Performs a hyperparameter grid search for DBSCAN.

    Parameters:
        * X_data            = data used to fit the DBSCAN instance
        * lst               = a list to store the results of the grid search
        * eps_space         = the range values for the eps parameter
        * min_samples_space = the range values for the min_samples parameter
        * min_clust         = the minimum number of clusters required after each search iteration in order for a result to be appended to the lst
        * max_clust         = the maximum number of clusters required after each search iteration in order for a result to be appended to the lst


    Example:

    # Loading Libraries
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    # Loading iris dataset
    iris = datasets.load_iris()
    X = iris.data[:, :]
    y = iris.target

    # Scaling X data
    dbscan_scaler = StandardScaler()

    dbscan_scaler.fit(X)

    dbscan_X_scaled = dbscan_scaler.transform(X)

    # Setting empty lists in global environment
    dbscan_clusters = []


    # Inputting function parameters
    dbscan_grid_search(X_data = dbscan_X_scaled,
                       lst = dbscan_clusters,
                       eps_space = pd.np.arange(0.1, 5, 0.1),
                       min_samples_space = pd.np.arange(1, 50, 1),
                       min_clust = 3,
                       max_clust = 6)

    """

    # Starting a tally of total iterations
    n_iterations = 0

    maxlen = 5

    # Looping over each combination of hyperparameters
    for eps_val in eps_space:

        clusters_tolerance = deque(maxlen=maxlen)
        noise_tolerance = deque(maxlen=maxlen)

        for samples_val in min_samples_space:

            dbscan_grid = DBSCAN(eps=eps_val, min_samples=samples_val,
                                 n_jobs=-1, metric='precomputed')

            # fit_transform
            dbscan_grid.fit(X=X_data)

            core_samples_mask = np.zeros_like(dbscan_grid.labels_, dtype=bool)
            core_samples_mask[dbscan_grid.core_sample_indices_] = True
            labels = dbscan_grid.labels_

            # Number of clusters in labels, ignoring noise if present.
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            n_noise_ratio = n_noise_ / len(labels)

            print('eps: {} samples: {} clusters: {} noise: {}'.format(
                eps_val, samples_val, n_clusters, n_noise_ratio))

            # Appending the lst each time n_clusters criteria is reached
            if min_clust <= n_clusters <= max_clust and n_noise_ratio <= noise_b:
                lst.append([eps_val, samples_val, n_clusters])
                print('append {} - {}'.format(n_clusters, n_noise_ratio))

            # Increasing the iteration tally with each run of the loop
            n_iterations += 1

            clusters_tolerance.append(n_clusters)
            noise_tolerance.append(n_noise_ratio)

            if len(clusters_tolerance) > maxlen:
                clusters_tolerance.popleft()

            if len(noise_tolerance) > maxlen:
                noise_tolerance.popleft()

            full = len(noise_tolerance) == len(noise_tolerance) == maxlen
            not_change = len(set(noise_tolerance)) == len(set(noise_tolerance)) == 1
            if full and not_change:
                print('break...')
                break

    # Printing grid search summary information
    print('Search Complete')
    print("Your list is now of length {}".format(len(lst)))
    print("Hyperparameter combinations checked: {}".format(n_iterations))
