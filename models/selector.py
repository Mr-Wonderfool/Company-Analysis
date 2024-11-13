from utils.UniVarSelector import UniVarSelector
from utils.RandomForestSelector import RandomForestSelector


class Selector:
    def __init__(self, data_X, data_Y):
        self.data_X = data_X
        self.data_Y = data_Y
        self.selector = None

    def select_univar(self, percent: int, method: str):
        """
        :param method: statistic selection method, in ['f_classif', 'mutual_info_classif']
        :returns: feature_f: the index of selected features
            idx: index of **all** features in descending order
        """
        assert method in ["f_classif", "mutual_info_classif"]
        self.selector = UniVarSelector(
            self.data_X, self.data_Y, percent=percent, method=method
        )
        features_f, idx = self.selector.fit()
        return features_f, idx

    def select_random_forest(self, num_trees: int):
        """
        :param method: random forest selection
        :returns: feature_f: the index of selected features
            idx: index of **all** features in descending order
        """
        self.selector = RandomForestSelector(self.data_X, self.data_Y, trees=num_trees)
        features_mutual_info, idx = self.selector.fit()
        return features_mutual_info, idx

    def plot(
        self,
        figsize,
        columns,
        threshold,
    ):
        return self.selector.plot(figsize=figsize, columns=columns, threshold=threshold)
