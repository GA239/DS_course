from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
from yellowbrick.text import TSNEVisualizer

from consts import TARGET_COLUMNS, TEXT_COLUMN
from my_utils import save_result, load


class Plotter:

    def __init__(self):
        super().__init__()

    @save_result('-', calculate=False, skip=True)
    def clusters_word_cloud(self, name: str, title: str = 'title'):
        """Creates Word Cloud picture fot mapping with name `name`"""
        mapping = load(name)
        pdf = pd.DataFrame(self.data.groupby(name).agg(TEXT_COLUMN).sum())

        fig = plt.figure(figsize=(10, 10))
        for i in range(len(mapping)):
            ax = fig.add_subplot(7, 3, i + 1)
            word_cloud = WordCloud().generate(pdf['data'][i])
            ax.set_title("WordCloud " + mapping[pdf.index[i]])
            ax.imshow(word_cloud, interpolation='bilinear')
            ax.axis('off')
        fig.suptitle(title)
        return fig

    @save_result('-', calculate=False, skip=False)
    def plot_umap(self, num: int = 2, title: str = 'title', target: bool = True):
        """
        Creates UMAP plot for umap_embendings.

        target flag to choose real labels of predicted ones
        """
        sns.set(style='white')

        # mixin
        standard_embedding = self.get_umap_embendings()
        _labels = self.get_target(num) if target else self.get_labels(num)

        fig = plt.figure(figsize=(20, 15))

        clustered = (_labels >= 0)
        plt.scatter(standard_embedding[~clustered, 0],
                    standard_embedding[~clustered, 1],
                    c=(0.5, 0.5, 0.5),
                    s=0.1,
                    alpha=0.5)

        plt.scatter(standard_embedding[clustered, 0],
                    standard_embedding[clustered, 1],
                    c=_labels[clustered],
                    s=0.1,
                    cmap='Spectral')

        plt.xlabel('true label')
        plt.axis('off')
        fig.suptitle("{}_{}".format(title, num))
        return fig

    @save_result('-', calculate=False, skip=True)
    def clusters_tsne(self, labels: pd.Series, title: str = 'title'):
        tsne = TSNEVisualizer(random_state=42)
        tsne.fit(self.vectors, labels)
        f = tsne.show().figure
        f.set_figheight(15)
        f.set_figwidth(15)
        f.suptitle(title)
        return f

    @save_result('-', calculate=False, skip=False)
    def plot_confusion_matrix(self, truth, k_labels, title='title'):
        fig = plt.figure(figsize=(8, 8))
        # Compute confusion matrix
        mat = metrics.confusion_matrix(truth, k_labels)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=set(truth),
                    yticklabels=set(truth))
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        fig.suptitle(title)
        return fig

    @staticmethod
    @save_result('-', calculate=False, skip=True)
    def plot_linkage_matrix(linkage_matrix, ylimit=None, title="ward_clusters", truncate_mode=None, p=5):
        fig, ax = plt.subplots(figsize=(15, 20))

        kwargs = {
            'leaf_rotation': 90.,  # rotates the x axis labels
            'leaf_font_size': 8.,  # font size for the x axis labels
            'show_contracted': False,  # to get a distribution impression in truncated branches
            'show_leaf_counts': False,  # otherwise numbers in brackets are counts

        }
        if truncate_mode:
            kwargs.update({
                'truncate_mode': truncate_mode,  # show only the last p merged clusters
                'p': p,  # show only the last p merged clusters
            })

        axs = dendrogram(linkage_matrix, **kwargs);

        if ylimit:
            ax.set_ylim(*ylimit)

        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off')

        plt.tight_layout();  # show plot with tight layout
        fig.suptitle(title)
        return fig

    def plot_clusters_scores(self):
        fig, ax = plt.subplots()
        v = self.calculate_number_of_clusters()
        ax.plot(v.k_values_, v.k_scores_, '*k:')
        ax.plot([
            v.k_values_[16],
            v.k_values_[12],
            v.k_values_[3]], [v.k_scores_[16],
                              v.k_scores_[12],
                              v.k_scores_[3]], 'or')
        ax.grid(True)

        ax.set(xlabel='k values ', ylabel='k scores',
               title='number of clusters')
        fig.set_figheight(15)
        fig.set_figwidth(15)
        plt.show()

    @save_result('-', calculate=False, skip=True)
    def plot_clusters(self, func, title='title'):
        fig, axs = plt.subplots(1, 3, figsize=(19, 3))
        for num, col in enumerate(TARGET_COLUMNS):
            self.plot(axs[num], self.convert_params(*func(self.data, col)))
        fig.suptitle(title)
        return fig


class PlotterBar(Plotter):

    @staticmethod
    def convert_params(x, y):
        return {"x": x, "height": y}

    def plot(self, ax, params):
        ax.bar(**params)


class PlotterScatter(Plotter):

    def __init__(self):
        super().__init__()

    @staticmethod
    def convert_params(x, y, z):
        return {'x': x, 'y': y, 'c': z}

    def plot(self, ax, params):
        ax.scatter(**params)

    def plot_clusters_svd(self, func):
        svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
        svd_df = pd.DataFrame(data=svd.fit_transform(self.vectors), index=self.data.index, columns=["x", "y"])
        self.plot_clusters(partial(func, svd_df=svd_df), title='scatter plot for clusters SVD')

    def plot_clusters_svd_deep(self, func, with_text_features=True, n_components=300):
        title = 'scatter plot for clusters SVD & then PCA {} {}'.format(with_text_features, n_components)
        svd = TruncatedSVD(n_components=15, n_iter=7, random_state=42)
        features = svd.fit_transform(self.vectors)

        if with_text_features:
            tmp = self.data[['word_count', 'length', 'word_density', 'compound', 'neg', 'neu', 'pos']]
            tmp = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(tmp)
            tmp = StandardScaler().fit_transform(tmp)
            features = np.concatenate([np.array(tmp), features], axis=1)

        pca = PCA(n_components=2, random_state=42)
        svd_df = pd.DataFrame(data=pca.fit_transform(features), index=self.data.index, columns=["x", "y"])
        self.plot_clusters(partial(func, svd_df=svd_df), title=title)
