import numpy as np
import math
from collections import defaultdict, Counter
from math import log as logarithm
from operator import itemgetter
import queue
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def mean(xs):
    return float(sum(xs)) / len(xs)


def isnan(val):
    return type(val) == float and math.isnan(val)


isnan = np.vectorize(isnan)


class C45:
    def __init__(self, max_depth=float("inf"), min_gain=0, continuous={}, depth=0):
        """
        Arguments:
            max_depth: After eaching this depth, the current node is turned into a leaf which predicts
                the most common label. This limits the capacity of the classifier and helps combat overfitting
            min_gain: The minimum gain a split has to yield. Again, this helps overfitting
            depth: Let's the current node know how deep it is into the tree, users usually don't need to set this
        """

        self.depth = depth
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.continuous = continuous

        self.leaf = False
        self.value = None

        self.children = {}
        self.feature = 0
        self.feature_split = None

    def fit(self, X, y):
        """
        Creates a tree structure based on the passed data

        Arguments:
            X: numpy array that contains the features in its rows
            y: numpy array that contains the respective labels
        """

        self.counts = Counter(y)
        self.most_common_label = self.counts.most_common()[0][0]

        # If there is only one class left, turn this node into a leaf
        # and always return this one value
        if len(set(y)) == 1:
            self.leaf = True
            self.value = y[0]
        # If the tree is getting to deep, turn this node into a leaf
        # and always predict the most common value
        elif self.depth >= self.max_depth:
            self.leaf = True
            self.value = self.most_common_label
        elif len({tuple(row) for row in X}) == 1:
            self.leaf = True
            self.value = self.most_common_label
        # Otherwise, look for the most informative feature and do a split on its possible values
        else:
            self.feature, self.feature_split = self._choose_feature(X, y)

            # If no feature is informative enough, turn this node into a leaf
            # and always predict the most common value
            if self.feature is None:
                self.leaf = True
                self.value = self.most_common_label
            else:
                if self.feature in self.continuous:
                    partition = self._partition_continuous(X, y, self.feature, self.feature_split)
                else:
                    partition = self._partition(X, y, self.feature)

                if self._is_useful_partition(partition):
                    self._save_partition_proportions(partition)

                    for value, (Xi, yi) in partition.items():
                        child = C45(continuous=self.continuous, depth=self.depth + 1, max_depth=self.max_depth)
                        child.fit(Xi, yi)
                        self.children[value] = child
                else:
                    self.leaf = True
                    self.value = self.most_common_label

    def predict_single(self, x):
        """
        Predict the class of a single data point x by either using the value encoded in a leaf
        or by following the tree structure recursively until a leaf is reached

        Arguments:
            x: individual data point
        """

        if self.leaf:
            return self.value
        else:
            value = x[self.feature]

            if isnan(value):
                return self._get_random_child_node().predict_single(x)
            elif self.feature in self.continuous:
                return self._predict_single_continuous(x, value)
            else:
                return self._predict_single_discrete(x, value)

    def _predict_single_discrete(self, x, value):
        if value in self.children:
            return self.children[value].predict_single(x)
        else:
            return self.most_common_label

    def _predict_single_continuous(self, x, value):
        if value <= self.feature_split:
            node = "smaller"
        else:
            node = "greater"

        return self.children[node].predict_single(x)

    def predict(self, X):
        """
        Predict the results for an entire dataset

        Arguments:
            X: numpy array that contains each data point in a row
        """

        return [self.predict_single(x) for x in X]

    def score(self, X, y):
        """
        Returns the accuracy for predicting the given dataset X
        """

        correct = sum(self.predict(X) == y)
        return float(correct) / len(y)

    def _choose_feature(self, X, y):
        """
        Finds the most informative feature to split on and returns its index.
        If no feature is informative enough, `None` is returned
        """

        best_feature = 0
        best_feature_gain = -float("inf")
        best_feature_split = None

        for i in range(X.shape[1]):
            gain, split = self._information_gain(X, y, i)

            if gain > best_feature_gain:
                best_feature = i
                best_feature_gain = gain
                best_feature_split = split

        if best_feature_gain < self.min_gain:
            best_feature = None

        self.gain = best_feature_gain

        return best_feature, best_feature_split

    def _information_gain(self, X, y, feature):
        if feature in self.continuous:
            max_gain, best_split = self._information_gain_continuous(X, y, feature)
            return max_gain, best_split
        else:
            return self._information_gain_discrete(X, y, feature), 0

    def _information_gain_continuous(self, X, y, feature):
        """
        Calculates the information gain achieved by splitting on the given feature
        """

        data, splits = self._get_continuous_splits(X, y, feature)

        old_entropy = self._entropy(y)

        max_gain = -float("inf")
        best_split = None

        for split in splits:
            smaller = [yi for (xi, yi) in data if xi <= split]
            greater = [yi for (xi, yi) in data if xi > split]

            ratio_smaller = float(len(smaller)) / len(data)

            new_entropy = ratio_smaller * self._entropy(smaller) + (1 - ratio_smaller) * self._entropy(greater)

            result = old_entropy - new_entropy

            if result > max_gain:
                best_split = split
                max_gain = result

        return max_gain, best_split

    def _information_gain_discrete(self, X, y, feature):
        """
        Calculates the information gain achieved by splitting on the given feature
        """

        result = self._entropy(y)

        summed = 0

        for value, (Xi, yi) in self._partition(X, y, feature).items():
            # Missing values should be ignored for computing the entropy
            if not isnan(value):
                summed += float(len(yi)) / len(y) * self._entropy(yi)

        result -= summed

        return result

    def _entropy(self, X):
        """
        Calculates the Shannon entropy on the given data X

        Arguments:
            X: An iterable for feature values. Usually, this is now a 1D list
        """

        summed = 0
        counter = Counter(X)

        for value in counter:
            count = counter[value]
            px = count / float(len(X))
            summed += px * logarithm(1. / px, 2)

        return summed

    def _partition(self, X, y, feature):
        """
        Partitioning is a common operation needed for decision trees (or search trees).
        Here, a partitioning is represented by a dictionary. The keys are values that the feature
        can take. Under each key, we save a tuple (Xi, yi) that represents all data points (and their labels)
        that have the respective value in the specified feature.
        """

        partition = defaultdict(lambda: ([], []))

        for Xi, yi in zip(X, y):
            bucket = Xi[feature]

            partition[bucket][0].append(Xi)
            partition[bucket][1].append(yi)

        partition = dict(partition)

        for feature, (Xi, yi) in partition.items():
            partition[feature] = (np.array(Xi), np.array(yi))

        return partition

    def _partition_continuous(self, X, y, feature, split):
        xi = X[:, feature]
        smaller = xi <= split
        greater = xi > split
        unknown = isnan(xi)

        ratio_smaller = sum(smaller) / float(sum(smaller) + sum(greater))

        unknown = np.where(unknown)[0]
        np.random.shuffle(unknown)

        greater[unknown] = True

        num_smaller = sum(smaller)
        num_greater = sum(greater)

        partition = {
            "smaller": (X[smaller], y[smaller]),
            "greater": (X[greater], y[greater])
        }

        return partition

    def _get_continuous_splits(self, X, y, feature):
        yi = y
        xi = X[:, feature]

        datai = sorted(zip(xi, yi), key=itemgetter(0, 1))

        splits = []

        xs = []
        ys = []
        last_x = None

        for xj, yj in datai:
            # Missing values can't be used to find good thresholds
            if isnan(xj):
                continue

            if xj == last_x:
                xs[-1].append(xj)
                ys[-1].add(yj)
            else:
                xs.append([xj])
                ys.append({yj})

            last_x = xj

        last_label = None

        for xj, yj in zip(xs, ys):
            if len(yj) == 1 and list(yj)[0] == last_label:
                splits[-1] += xj
            else:
                splits.append(xj)

            if len(yj) == 1:
                last_label = list(yj)[0]
            else:
                last_label = None

        splits = [mean(vals) for vals in splits]

        return datai, splits

    def _is_useful_partition(self, partition):
        num_useful = 0

        for value, (Xi, yi) in partition.items():
            if len(yi) > 0:
                num_useful += 1

        return num_useful >= 2

    def _save_partition_proportions(self, partition):
        occurences = {}

        for child, (xj, xi) in partition.items():
            occurences[child] = len(xj)

        total = float(sum(occurences.values()))

        self.children_probs = {child: occ / total for child, occ in occurences.items()}

    def _get_random_child_node(self):
        name = np.random.choice(self.children_probs.keys(), p=self.children_probs.values())
        return self.children[name]

    def prune(self, X_val, y_val):
        old_score = self.score(X_val, y_val)
        pruned = 0
        not_pruned = 0

        for node in reversed(self._bfs()):
            if not node.leaf:
                node._make_leaf()
                score = self.score(X_val, y_val)

                if old_score > score:
                    node._make_internal()
                    not_pruned += 1
                else:
                    old_score = score
                    pruned += 1

        return not_pruned, pruned

    def _bfs(self):
        que = queue.Queue()
        node = self

        que.put(node)
        result = [node]

        while not que.empty():
            if not node.leaf:
                for _, child in node.children.items():
                    que.put(child)

            node = que.get()
            result.append(node)

        return result

    def _make_leaf(self):
        self.leaf = True
        self.value = self.most_common_label

    def _make_internal(self):
        self.leaf = False
        self.value = None


data = DataFrame.from_csv("/home/mario/IML/titanic_modified.csv")
y = data["Survived"].as_matrix()

data.info()


def preprocess(data, encode_labels=False):
    X = data.drop(["Survived"], 1)
    if encode_labels:  # for sklearn
        for x in X:
            X[x] = LabelEncoder().fit_transform(X[x].astype(str))

    print(X.head(10))

    X = X.as_matrix()

    return X


X = preprocess(data, encode_labels=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = C45(continuous={1})
clf.fit(X_train, y_train)

print("train accuracy = %.5f" % clf.score(X_train, y_train))
print("test accuracy = %.5f" % clf.score(X_test, y_test))