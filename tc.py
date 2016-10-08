import nltk
from collections import Counter
import pandas as pd
import string
import numpy as np
import sklearn
import re


def process(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ Normalizes case and handles punctuation
    Inputs:
        text: str: raw text
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs:
        list(str): tokenized text
    """
    regex = re.compile("[{0}\s]+".format(re.escape(string.punctuation)))
    tokens = re.split(regex, string.lower(text).replace("'s", "").replace("'", ""))
    result = []
    for token in tokens:
        try:
            if len(token) > 0:
                result.append(lemmatizer.lemmatize(token))
        except:
            pass
    return result


def process_all(df, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ process all text in the dataframe using process_text() function.
    Inputs
        df: pd.DataFrame: dataframe containing a column 'text' loaded from the CSV file
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs
        pd.DataFrame: dataframe in which the values of text column have been changed from str to list(str),
                        the output from process_text() function. Other columns are unaffected.
    """
    df["text"] = df["text"].map(lambda x: process(x, lemmatizer))
    return df


def get_rare_words(processed_tweets):
    """ use the word count information across all tweets in training data to come up with a feature list
    Inputs:
        processed_tweets: pd.DataFrame: the output of process_all() function
    Outputs:
        list(str): list of rare words, sorted alphabetically.
    """
    cnt = Counter()
    for text in processed_tweets["text"]:
        for term in text:
            cnt[term] += 1
    return [term for term, value in cnt.most_common() if value == 1]


def create_features(processed_tweets, rare_words):
    """ creates the feature matrix using the processed tweet text
    Inputs:
        tweets: pd.DataFrame: tweets read from train/test csv file, containing the column 'text'
        rare_words: list(str): one of the outputs of get_feature_and_rare_words() function
    Outputs:
        sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used
                                                we need this to tranform test tweets in the same way as train tweets
        scipy.sparse.csr.csr_matrix: sparse bag-of-words TF-IDF feature matrix
    """
    vocab = set()
    docs = []
    stopwords = set(rare_words + nltk.corpus.stopwords.words('english'))
    for terms in processed_tweets["text"]:
        doc = []
        for term in terms:
            if term not in stopwords:
                vocab.add(term)
                doc.append(term)
        docs.append(" ".join(doc))
    model = sklearn.feature_extraction.text.TfidfVectorizer(vocabulary=vocab).fit(docs)
    return model, model.transform(docs)


def get_label(name, label_set):
    return 0 if name in label_set else 1


def create_labels(processed_tweets):
    """ creates the class labels from screen_name
    Inputs:
        tweets: pd.DataFrame: tweets read from train file, containing the column 'screen_name'
    Outputs:
        numpy.ndarray(int): dense binary numpy array of class labels
    """
    label_set = {"realDonaldTrump", "mike_pence", "GOP"}
    return np.array([get_label(name, label_set) for name in processed_tweets["screen_name"]])


def learn_classifier(X_train, y_train, kernel='best'):
    """ learns a classifier from the input features and labels using the kernel function supplied
    Inputs:
        X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features, output of create_features_and_labels()
        y_train: numpy.ndarray(int): dense binary vector of class labels, output of create_features_and_labels()
        kernel: str: kernel function to be used with classifier. [best|linear|poly|rbf|sigmoid]
                    if 'best' is supplied, reset the kernel parameter to the value you have determined to be the best
    Outputs:
        sklearn.svm.classes.SVC: classifier learnt from data
    """
    if kernel == 'best':
        kernel = 'linear'  # fill the best mode after you finish the evaluate_classifier() function
    clf = sklearn.svm.classes.SVC(kernel=kernel)
    clf = clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(classifier, X_validation, y_validation):
    """ evaluates a classifier based on a supplied validation data
    Inputs:
        classifier: sklearn.svm.classes.SVC: classifer to evaluate
        X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features
        y_train: numpy.ndarray(int): dense binary vector of class labels
    Outputs:
        double: accuracy of classifier on the validation data
    """
    return classifier.score(X_validation, y_validation)


def classify_tweets(tfidf, classifier, unlabeled_tweets):
    """ predicts class labels for raw tweet text
    Inputs:
        tfidf: sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used on training data
        classifier: sklearn.svm.classes.SVC: classifier learnt
        unlabeled_tweets: pd.DataFrame: tweets read from tweets_test.csv
    Outputs:
        numpy.ndarray(int): dense binary vector of class labels for unlabeled tweets
    """
    processed_tweets = process_all(unlabeled_tweets)
    docs = []
    for terms in processed_tweets["text"]:
        docs.append(" ".join(terms))
    x_test = tfidf.transform(docs)
    return classifier.predict(x_test)



# AUTOLAB_IGNORE_START
text = "This'saa a sam$ple test input for processing."
print process(text)
# AUTOLAB_IGNORE_STOP

# AUTOLAB_IGNORE_START
tweets = pd.read_csv("tweets_train.csv", na_filter=False)
print tweets.head()
# AUTOLAB_IGNORE_STOP

# AUTOLAB_IGNORE_START
processed_tweets = process_all(tweets)
print processed_tweets.head()
# AUTOLAB_IGNORE_STOP

# AUTOLAB_IGNORE_START
rare_words = get_rare_words(processed_tweets)
print len(rare_words) # should give 19623
# AUTOLAB_IGNORE_STOP

# AUTOLAB_IGNORE_START
(tfidf, X) = create_features(processed_tweets, rare_words)
print type(tfidf)
print type(X)
# AUTOLAB_IGNORE_STOP

# AUTOLAB_IGNORE_START
y = create_labels(processed_tweets)
print y
# AUTOLAB_IGNORE_STOP


# AUTOLAB_IGNORE_START
classifier = learn_classifier(X, y, 'linear')
# AUTOLAB_IGNORE_STOP


# AUTOLAB_IGNORE_START
accuracy = evaluate_classifier(classifier, X, y)
print accuracy # should give 0.954850271708

for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
    classifier = learn_classifier(X, y, kernel)
    accuracy = evaluate_classifier(classifier, X, y)
    print kernel,':',accuracy

# AUTOLAB_IGNORE_STOP


# AUTOLAB_IGNORE_START
classifier = learn_classifier(X, y, 'best')
unlabeled_tweets = pd.read_csv("tweets_test.csv", na_filter=False)
y_pred = classify_tweets(tfidf, classifier, unlabeled_tweets)
print type(y_pred)
# AUTOLAB_IGNORE_STOP