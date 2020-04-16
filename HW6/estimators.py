import abc
from collections import Counter

import hdbscan
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans
from sklearn.utils import shuffle
from yellowbrick.cluster import KElbowVisualizer

from my_utils import load, save_result, dbscan_grid_search


class Estimator(abc.ABC):

    @abc.abstractmethod
    def estimator_name(self):
        pass

    @abc.abstractmethod
    def get_model(self):
        pass

    def show_metrics(self, truth, k_labels):
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(truth, k_labels))
        print("Completeness: %0.3f" % metrics.completeness_score(truth, k_labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(truth, k_labels))
        print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(truth, k_labels))
        print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(truth, k_labels))

    @save_result('number_of_clusters_visualizer', calculate=False, skip=False)
    def calculate_number_of_clusters(self):
        visualizer = KElbowVisualizer(self.get_model()(), k=(4, 30))
        visualizer.fit(self.vectors)  # Fit the data to the visualizer
        return visualizer


class KMeansEstimator(Estimator):

    def estimator_name(self):
        return 'kmenas'

    def get_model(self):
        return KMeans

    @save_result('-', calculate=False, skip=False)
    def calculate(self, num_clusters, file_name=None):
        return self.get_model()(n_clusters=num_clusters).fit_predict(self.vectors)

    def get_labels(self, num):
        num_clus = len(set(self.get_target(num).values))
        return self.calculate(num_clus, file_name='{}_{}'.format(self.estimator_name(), num_clus))

    def calculate_metrics(self, num):
        truth = self.get_target(num).values
        k_labels = self.get_labels(num)
        self.show_metrics(truth, k_labels)
        # num_clus = len(set(truth))
        # self.plot_confusion_matrix(truth, k_labels,
        #                            title='{}_{}_confusion_matrix'.format(self.estimator_name(), num_clus))


class DBSCANEstimator(Estimator):

    def __init__(self, *args, **kwargs):
        super(Estimator, self).__init__(*args, **kwargs)
        self._params = {
            'eps': 0.8,
            'min_samples': 10,
            'n_jobs': -1,
            'metric': 'precomputed'
        }

    def estimator_name(self):
        return 'DBSCAN_eps_{}_min_samples_{}'.format(self.params()['eps'],
                                                     self.params()['min_samples'])

    def get_model(self):
        return DBSCAN

    def params(self):
        return self._params

    def set_params(self, param_dict):  # TODO rewrite to property setter/getter
        self._params.update(param_dict)

    @property
    def scan_vectors(self):
        return self.cosine_dist

    @save_result('-', calculate=False, skip=False)
    def calculate(self, file_name=None):
        db = self.get_model()(**self.params()).fit(self.scan_vectors)
        labels = db.labels_

        print(Counter(labels))
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print('Estimated number of clusters_: %d' % n_clusters_)
        print('Estimated number of points: %d' % len(labels))
        print('Estimated number of noise points: %d' % n_noise_)

        return labels

    def get_labels(self, num=1):
        return self.calculate(file_name=self.estimator_name())

    def calculate_metrics(self):
        k_labels = self.get_labels(None)
        truth = self.get_target(2).values
        self.show_metrics(truth, k_labels)

    #         self.plot_confusion_matrix(truth, k_labels,
    #                                    title='{}_{}_confusion_matrix'.format(
    #                                        self.estimator_name(), len(set(truth))));

    @save_result('-', calculate=False, skip=False)
    def Dbscan_CV(self, eps_space, min_samples_space, file_name=None):
        dbscan_clusters = []
        # Inputting function parameters
        dbscan_grid_search(X_data=self.cosine_dist,
                           lst=dbscan_clusters,
                           eps_space=eps_space,
                           min_samples_space=min_samples_space,
                           min_clust=6,
                           max_clust=21)
        return dbscan_clusters

    @staticmethod
    def params_generator(self):
        cv_pd = pd.DataFrame(np.array(load("dbscan_clusters_second")),
                             columns=['eps', 'min_samples', 'num_clusters'])
        cv_pd = shuffle(cv_pd).drop_duplicates('num_clusters')[['eps', 'min_samples']]
        cv_pd['args'] = cv_pd[['eps', 'min_samples']].apply(lambda x: {
            'eps': x['eps'], 'min_samples': x['min_samples']}, axis=1)
        cv_pd = cv_pd.drop(columns=['eps', 'min_samples'])
        for param in cv_pd.to_dict()['args'].values():
            yield param


class HDBSCANEstimator(DBSCANEstimator):

    def __init__(self, *args, **kwargs):
        super(Estimator, self).__init__(*args, **kwargs)
        self._params = {
            'min_cluster_size': 100,
            'min_samples': 10,
            'metric': 'precomputed',
            'cluster_selection_epsilon': 0.8
        }

    def estimator_name(self):
        prms = self.params()
        return 'HDBSCAN_eps_{}_min_samples_{}_min_cluster_size_{}'.format(prms['cluster_selection_epsilon'],
                                                                          prms['min_samples'],
                                                                          prms['min_cluster_size'])

    def get_model(self):
        return hdbscan.HDBSCAN


class EnhancedHDBSCANEstimator(HDBSCANEstimator):

    def estimator_name(self):
        prms = self.params()
        return 'EnhancedHDBSCANEstimator_eps_{}_min_samples_{}_min_cluster_size_{}'.format(
            prms['cluster_selection_epsilon'],
            prms['min_samples'],
            prms['min_cluster_size'])

    @property
    def scan_vectors(self):
        return self.get_umap_embendings()
