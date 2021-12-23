import os
import json
import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import LabelPowerset


PARENT_PATH = os.path.join("..")
DATA_CSV = os.path.join(PARENT_PATH, "data", "prepared.csv")
TAGS_TXT = os.path.join(PARENT_PATH, "data", "tags.txt")
TAGS = []

def get_tags(file):
    with open(file, 'r') as fr:
        return json.load(fr)

def one_vs_rest(x_train, y_train, x_test, y_test):
    start_time = datetime.datetime.now()
    LogReg_pipeline = Pipeline([
                    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
                ])

    accuracy_all = []
    for tag in TAGS:
        LogReg_pipeline.fit(x_train, train[tag])
        prediction = LogReg_pipeline.predict(x_test)
        accuracy = accuracy_score(test[tag], prediction)
        accuracy_all.append(accuracy)

        print('Test accuracy of {} is {}'.format(tag, accuracy))
    print('Test (np.mean) accuracy is {}'.format(np.mean(accuracy_all)))
    final_time = datetime.datetime.now()
    print('Time: {}'.format(final_time-start_time))

def binary_relevance(x_train, y_train, x_test, y_test):
    classifier = BinaryRelevance(GaussianNB())

    start_time = datetime.datetime.now()
    classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_test)
    print('Accuracy is {}'.format(accuracy_score(y_test,predictions)))
    final_time = datetime.datetime.now()
    print('Time: {}'.format(final_time-start_time))


def classifier_chain(x_train, y_train, x_test, y_test):
    classifier = ClassifierChain(LogisticRegression())

    start_time = datetime.datetime.now()
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    print('Accuracy is {}'.format(accuracy_score(y_test,predictions)))
    final_time = datetime.datetime.now()
    print('Time: {}'.format(final_time-start_time))

def label_powerset(x_train, y_train, x_test, y_test):
    classifier = LabelPowerset(LogisticRegression())
    start_time = datetime.datetime.now()
    classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_test)
    print('Accuracy is {}'.format(accuracy_score(y_test,predictions)))
    final_time = datetime.datetime.now()
    print('Time: {}'.format(final_time-start_time))

if __name__ == "__main__":
    TAGS = get_tags(TAGS_TXT)

    df = pd.read_csv(DATA_CSV, index_col=0)
    df = df.dropna()

    train, test = train_test_split(df, test_size=0.2, random_state=0)

    train_text = train["text_stem"]
    test_text = test["text_stem"]

    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_text)
    vectorizer.fit(test_text)


    x_train = vectorizer.transform(train_text)
    y_train = train.drop(labels = ['text_stem','tags'], axis=1)

    x_test = vectorizer.transform(test_text)
    y_test = test.drop(labels = ['text_stem','tags'], axis=1)

    one_vs_rest(x_train, y_train, x_test, y_test)
    binary_relevance(x_train, y_train, x_test, y_test)
    classifier_chain(x_train, y_train, x_test, y_test)
    label_powerset(x_train, y_train, x_test, y_test)
