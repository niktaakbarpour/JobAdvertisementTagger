from src.Tagger import Tagger
from src.Utils import read_cleaned_excel, write_to_excel


def main():
    # Reading the dataset
    raw_data = read_cleaned_excel("../raw/CleanedDataset.xlsx")
    # Depending on the type of Tagging, one of the args below must be passed to tagger object
    tagger = Tagger()
    # Tagging with Regex
    args = {
        "type": "regex"
    }
    # Tagging with classification
    # It first uses Regex to set labels for as many ads as possible, then uses these ads as  a training set to train a
    # classification model, then classifies other ads with this model.
    args = {
        'type': "classification",
        'classifier': KNeighborsClassifier(n_neighbors=5, metric='euclidean'),  # Specify the classification model
        'selected_features': ['title', 'keyword'],  # Specify features that must be considered in classification
        'keyword_count_threshold': 5,  # Specify a minimum count for keywords to be involved in features
        'use_idf': True  # Specify whether IDF must be used in vectorization of titles
    }

    # Tagging with clustering
    args = {
        'type': "clustering",
        'cluster_count': 8,  # Specify the number of cluster
        'clusterer': KMeans(n_clusters=8, init='k-means++', random_state=42),  # Specify the clustering model
        'selected_features': ['title', 'keyword'],  # Specify features that must be considered in classification
        'keyword_count_threshold': 5,  # Specify a minimum count for keywords to be involved in features
        'use_idf': True  # Specify whether IDF must be used in vectorization of titles
    }

    # Tagging dataset
    tagged_data = tagger.fit_transform(raw_data, args)
    # Storing tagged data
    write_to_excel(tagged_data, "../out/TaggedData.xlsx")
    print("Done")


if __name__ == '__main__':
    main()
