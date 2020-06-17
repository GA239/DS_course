import gc

import pandas as pd
import umap
from scipy.cluster.hierarchy import ward
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

from consts import TARGET_COLUMNS, TEXT_COLUMN
from my_utils import save_result


class DataAdapter:

    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__()
        self.data = df.copy()

    @property
    def _text_data(self):
        """return raw text data"""
        return self.data[TEXT_COLUMN]

    def get_target(self, num: int) -> pd.DataFrame:
        """return labels according to num (1,2,3 - hierarchies levels)"""
        return self.data[TARGET_COLUMNS[num]]

    def update_data(self, df: pd.DataFrame) -> None:
        del self.data
        gc.collect()
        self.data = df.copy()


class TfidfVectoriserAdapter(DataAdapter):

    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)

        self.vectorizer = TfidfVectorizer()
        self.is_fited = False
        self._vectors = None

    @property
    def vectors(self, refit=False):
        if not self.is_fited or refit:
            self._vectors = self.vectorizer.fit_transform(self._text_data.tolist())
            self.is_fited = True
        return self._vectors

    @property
    def get_cosine_similarity(self):
        return cosine_similarity(self.vectors)

    @property
    @save_result('tfidf_cosine_dist', calculate=False, skip=False)
    def cosine_dist(self):
        return cosine_distances(self.vectors)

    @property
    @save_result('tfidf_wrap_cosdist', calculate=False, skip=False)
    def linkage_matrix(self):
        print('calculate linkage_matrix')
        return ward(self.cosine_dist)

    def illustrate_dendrogram(self, mode="Full"):
        vtype = "tfidf"
        if mode == "Full":
            self.plot_linkage_matrix(self.linkage_matrix, title="dendrogram_{}_full".format(vtype))
            return
        self.plot_linkage_matrix(self.linkage_matrix,
                                 ylimit=(1.2, None),
                                 truncate_mode='level', p=5,
                                 title="dendrogram_{}_truncate_mode_{}_p_{}_with_ylimit".format(vtype, 'level', 5))

    @save_result('umap_embendings', calculate=False, skip=False)
    def get_umap_embendings(self):
        return umap.UMAP(n_neighbors=20,
                         min_dist=0.0,
                         n_components=2,
                         metric='cosine',
                         random_state=42).fit_transform(self.vectors)

