import re
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

from src.KeywordsEncoder import KeywordsEncoder
from src.TitleEncoder import TitleEncoder


class Tagger:
    def __init__(self):
        self.data = pd.DataFrame()
        self.frontendMatcher = re.compile(
            "front([\\s\\-_])?(end)?|java([\\s\\-_])?script|react([\\s\\-_.])?(js)?|wordpress|word-press|\\bui\\b|\\bux\\b|\\buiux\\b|\\bvue([\\s\\-_.])?(js)?\\b|angular([\\s.\\-_])?(js)?")
        self.backendMatcher = re.compile(
            "back([\\s\\-_])?(end)?|\\basp\\b|\\b([\\-.]|dot)?net\\b|node([\\s\\-_.])?(js)?|\\bphp\\b|go([\\s\\-_])?(lang)?|laravel|django")
        self.mobileMatcher = re.compile("android|ios|mobile|flutter")
        self.aiMatcher = re.compile(
            "\\bai\\b|artificial([\\s\\-])?(intelligence)?|data(-|\\s)?scientist|business inteligence|\\bbi\\b|data(\\s)?(-)?(\\s)?(engineer|expert|export)")
        self.managingMatcher = re.compile(
            "\\bit\\b|manage|product|market(ing)?|scrum(\\s)?(master)?|leader|sal|\\bseo\\b|informatics|erp|advertis(e|ing)")
        self.networkMatcher = re.compile("network|arp|cisco")
        self.devopsMatcher = re.compile("devops|deploy(ment)?|devpps")
        self.databaseMatcher = re.compile("database|sql")
        self.fullstackMatcher = re.compile(
            "full(\\s)?([\\-_])?(\\s)?stack|web\\s(application\\s|applications\\s)?(programmer|developer|designer)")
        self.adminMatcher = re.compile(
            "admin|adminstration|host(ing)?|sysadmin|system admin|system adminstration|cloud|data(\\s)?center|server")
        self.securityMatcher = re.compile("security")
        self.osMatcher = re.compile("linux|windows|operating system|\\bos\\b")
        self.testMatcher = re.compile("test(ing)?")
        self.unused_keywords = self.read_unused_keywords("../raw/unused_keywords.txt")

    @staticmethod
    def read_unused_keywords(file_name: str) -> dict:
        unused_keywords = {}
        all_tags = open(file_name, encoding='utf-8', newline='\n').read().strip().split("\n")
        for pairs in all_tags:
            split = pairs.split(":")
            tag = split[0]
            keywords = set(split[1].split(","))
            unused_keywords[tag] = keywords
        return unused_keywords

    def fit(self, data: pd.DataFrame):
        self.data = data

    def transform(self, args: dict):
        tagger_type = args.get("type", 'regex')
        if tagger_type == 'regex':
            all_tags = []
            for title in self.data["JobTitle"]:
                title = title.lower()
                tags = []
                if re.search(self.frontendMatcher, title):
                    tags.append("FRONT_END")
                if re.search(self.backendMatcher, title):
                    tags.append("BACK_END")
                if re.search(self.mobileMatcher, title):
                    tags.append("MOBILE")
                if re.search(self.aiMatcher, title):
                    tags.append("AI")
                if re.search(self.managingMatcher, title):
                    tags.append("MANAGING")
                if re.search(self.networkMatcher, title):
                    tags.append("NETWORK")
                if re.search(self.devopsMatcher, title):
                    tags.append("DEVOPS")
                if re.search(self.databaseMatcher, title):
                    tags.append("DATABASE")
                if re.search(self.fullstackMatcher, title):
                    tags.append("FULL_STACK")
                if re.search(self.adminMatcher, title):
                    tags.append("SYS_ADMIN")
                if re.search(self.securityMatcher, title):
                    tags.append("SECURITY")
                if re.search(self.osMatcher, title):
                    tags.append("OS")
                if re.search(self.testMatcher, title):
                    tags.append("TEST")
                if len(tags) == 0:
                    tags = np.nan
                else:
                    tags = list(sorted(tags))
                all_tags.append(tags)
            self.data.insert(4, "Tag", all_tags, True)
        elif tagger_type == 'classification':
            data = self.transform({'type': 'regex'})
            if args.get("fix_multi_labels", False):
                data = self.fix_multi_labels(data)
            features = self.get_selected_features(data, args)
            x_train = features[~data.Tag.isna()]
            x_test = features[data.Tag.isna()]
            y_train = data[~data.Tag.isna()]["Tag"]
            y_train = list(map(lambda tag: ",".join(sorted(tag)), y_train))
            classifier = args.get('classifier', KNeighborsClassifier(n_neighbors=5, metric='euclidean'))
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            y_pred = list(map(lambda tag: tag.split(","), y_pred))
            data.loc[data.Tag.isna(), "Tag"] = y_pred
            self.data = data
        elif tagger_type == 'clustering':
            features = self.get_selected_features(self.data, args)
            cluster_count = args.get('cluster_count', 8)
            clustering = args.get('clusterer', KMeans(n_clusters=cluster_count, init='k-means++', random_state=42))
            y_pred = clustering.fit_predict(features)
            self.data.insert(4, "Tag", y_pred, True)
        else:
            raise ValueError("Illegal type")

        return self.data

    def fit_transform(self, data: pd.DataFrame, args: dict):
        self.fit(data)
        return self.transform(args)

    @staticmethod
    def get_selected_features(data: pd.DataFrame, args: dict):
        selected_feature_names = args.get('selected_features', ['title', 'keyword'])
        features = np.array([])
        if 'keyword' in selected_feature_names:
            threshold = args.get('keyword_count_threshold', None)
            encoded_keywords = KeywordsEncoder().fit_transform(data["Keywords"], threshold=threshold)
            features = encoded_keywords

        if 'title' in selected_feature_names:
            stop_words_path = args.get('stop_words_path', "../raw/ExtraStopWords.txt")
            max_features = args.get('max_title_features', None)
            use_idf = args.get('use_idf', True)
            encoded_titles = TitleEncoder(stop_words_path).fit_transform(data["JobTitle"],
                                                                         max_features=max_features,
                                                                         use_idf=use_idf)
            if len(features) > 0:
                features = np.append(features, encoded_titles, axis=1)
            else:
                features = encoded_titles
        return features

    @staticmethod
    def draw_wcss(X):
        wcss = []
        for i in range(1, 21):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 21), wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

    @staticmethod
    def calculate_k(data, max_size):
        num_clusters = np.arange(2, max_size + 1)
        results = {}
        for size in num_clusters:
            model = KMeans(n_clusters=size).fit(data)
            predictions = model.predict(data)
            results[size] = silhouette_score(data, predictions)

        best_size = max(results, key=results.get)
        return best_size

    @staticmethod
    def fix_multi_labels(data: pd.DataFrame) -> pd.DataFrame:
        deleted_index = []
        added_row = pd.DataFrame()
        for index in range(len(data)):
            row_tags = data.iloc[index]["Tag"]
            if type(row_tags) != float and len(row_tags) > 1:
                deleted_index.append(data.index[index])
                row = data.iloc[index]
                for tag in row_tags:
                    row.Tag = [tag]
                    added_row = added_row.append(row, ignore_index=True)
        data.drop(deleted_index, inplace=True)
        data = data.append(added_row, ignore_index=True)
        data.reset_index(drop=True, inplace=True)
        data["Tag"] = data["Tag"].apply(lambda tags: "OTHER" if pd.isna(tags) else ",".join(tags))
        return data

    def get_keywords_per_tag(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[~data.Tag.isna()]
        keywords = {}
        for index in range(len(data)):
            row_keyword = list(data.iloc[index]["Keywords"])
            if "ALL" not in keywords:
                keywords["ALL"] = []
            keywords.get("ALL").extend(row_keyword)
            tags = data.iloc[index]["Tag"]
            for tag in tags:
                if tag not in keywords:
                    keywords[tag] = []
                keywords.get(tag).extend(self.filter_unused_keyword_per_tag(tag, row_keyword))
        dataframe = pd.concat([pd.DataFrame({tag: keywords.get(tag)}) for tag in keywords], ignore_index=True, axis=1)
        dataframe.columns = keywords.keys()
        dataframe.reset_index(drop=True, inplace=True)
        return dataframe

    def filter_unused_keyword_per_tag(self, tag: str, keywords: set) -> set:
        return set(filter(lambda keyword: keyword not in self.unused_keywords.get(tag), keywords))
