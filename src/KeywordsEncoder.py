import numpy as np


class KeywordsEncoder:
    def __init__(self):
        self.data = []
        self.all_keywords = []

    def fit(self, data, threshold: int = None):
        self.data = data
        all_keywords = {}
        for keywords in data:
            for keyword in keywords:
                all_keywords[keyword] = all_keywords.get(keyword, 0) + 1
        if threshold is not None:
            keywords = set(all_keywords.keys())
            for keyword in keywords:
                if all_keywords.get(keyword) <= threshold:
                    all_keywords.pop(keyword)
        self.all_keywords = sorted(all_keywords.keys())

    def transform(self):
        encoded = []
        all_keyword_size = len(self.all_keywords)
        for row_keywords in self.data:
            row = [0] * all_keyword_size
            for row_keyword in row_keywords:
                try:
                    index = self.all_keywords.index(row_keyword)
                    row[index] = 1
                except ValueError:
                    pass
            encoded.append(row)
        return np.array(encoded)

    def fit_transform(self, data, threshold: int = None):
        self.fit(data, threshold)
        return self.transform()
