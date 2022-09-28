import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class TitleEncoder:
    def __init__(self, extra_stop_words_path: str = None):
        self.data = []
        nltk.download('stopwords')
        self.stopwords = stopwords.words('english')
        if extra_stop_words_path is not None:
            extra_stop_words = open(extra_stop_words_path, encoding='utf-8', newline='\n').read().strip().split("\n")
            extra_stop_words = list(filter(lambda stop_word: stop_word, extra_stop_words))
            self.stopwords.extend(extra_stop_words)

    def fit(self, data):
        self.data = data

    def transform(self, max_features=None, use_idf=True):
        corpus = []
        ps = PorterStemmer()
        for text in self.data:
            words = re.sub('[^a-zA-Z]', ' ', text).lower().split()
            new_words = []
            for word in words:
                if word not in self.stopwords:
                    stemmed = ps.stem(word)
                    if stemmed not in self.stopwords:
                        new_words.append(stemmed)
            text = ' '.join(new_words)
            corpus.append(text)
        transformer = TfidfVectorizer(max_features=max_features, use_idf=use_idf)
        return np.array(transformer.fit_transform(corpus).toarray())

    def fit_transform(self, data, max_features=None, use_idf=True):
        self.fit(data)
        return self.transform(max_features, use_idf)
