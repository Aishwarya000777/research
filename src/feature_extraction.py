
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_tfidf_features(train_texts, test_texts):

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        stop_words=None,
        min_df=1,
        max_features=None
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    print("\nTF-IDF feature extraction completed!")
    print("Training matrix:", X_train.shape)
    print("Test matrix:", X_test.shape)

    return X_train, X_test, vectorizer