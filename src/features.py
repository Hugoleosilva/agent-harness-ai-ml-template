"""Feature layer: statistics are fit on TRAIN only, then applied to any split."""


class Standardizer:
    def fit(self, x_train):
        self.mean_ = x_train.mean(axis=0)
        self.std_ = x_train.std(axis=0) + 1e-8
        return self

    def transform(self, x):
        return (x - self.mean_) / self.std_
