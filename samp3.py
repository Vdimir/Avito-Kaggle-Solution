# coding: utf-8
"""
Benchmarks for the Avito fraud detection competition
"""
import csv
import re
import nltk.corpus
from collections import defaultdict
import scipy.sparse as sp
import numpy as np
import os

from time import strftime

from sklearn.linear_model import SGDClassifier
from nltk import SnowballStemmer
import random as rnd
import logging
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score

from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm


import pandas as pd


stopwords = frozenset(word for word in nltk.corpus.stopwords.words("russian") if word != u"не")
# stopwords= frozenset( )
stemmer = SnowballStemmer('russian')
engChars = [ord(char) for char in u"cCyoOBaAKpPeE"]
rusChars = [ord(char) for char in u"сСуоОВаАКрРеЕ"]
eng_rusTranslateTable = dict(zip(engChars, rusChars))
rus_engTranslateTable = dict(zip(rusChars, engChars))

logging.basicConfig(format=u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level=logging.NOTSET)


def correctWord(w):
    """ Corrects word by replacing characters with written similarly depending on which language the word.
        Fraudsters use this technique to avoid detection by anti-fraud algorithms."""

    if len(re.findall(u"[а-я]", w)) > len(re.findall(u"[a-z]", w)):
        return w.translate(eng_rusTranslateTable)
    else:
        return w.translate(rus_engTranslateTable)

def getWords(text, stemmRequired=False, correctWordRequired=False):
    """ Splits the text into words, discards stop words and applies stemmer.
    Parameters
    ----------
    text : str - initial string
    stemmRequired : bool - flag whether stemming required
    correctWordRequired : bool - flag whether correction of words required
    """

    cleanText = re.sub(u'[^a-zа-я0-9]', ' ', text.lower())
    if correctWordRequired:
        words = [correctWord(w) if not stemmRequired or re.search("[0-9a-z]", w) else stemmer.stem(correctWord(w)) for w
                 in cleanText.split() if len(w) > 1 and w not in stopwords]
    else:
        words = [w if not stemmRequired or re.search("[0-9a-z]", w) else stemmer.stem(w) for w in cleanText.split() if
                 len(w) > 1 and w not in stopwords]

    return words


def processData(data, featureIndexes={}):
    """ Processing data. """
    processMessage = (
                     "Generate features for " if featureIndexes else "Generate features dict from ")
    logging.info(processMessage + "...")

    wordCounts = defaultdict(lambda: 0)
    targets = []
    item_ids = []
    row = []
    col = []
    cur_row = 0

    skipped = 0
    for processedCnt, item in data.iterrows():
        # col = []
        # if "is_blocked" in item:
        #     if (int(item["is_blocked"]) == 1 and int(item["is_proved"]) == 0):
        #         skipped += 1
        #         continue

        for word in getWords(str(item["title"]) + " " + str(item["description"]), stemmRequired=False, correctWordRequired=False):
            if not featureIndexes:
                wordCounts[word] += 1
            else:
                if word in featureIndexes:
                    col.append(featureIndexes[word])
                    row.append(cur_row)

        if featureIndexes:
            cur_row += 1
            if "is_blocked" in item:
                targets.append(int(item["is_blocked"]))
            item_ids.append(int(item["itemid"]))

        if processedCnt % 1000 == 0:
            logging.debug(processMessage + ": " + str(processedCnt) + " items done")

    logging.info("skip..")
    logging.info(skipped)
    if not featureIndexes:
        index = 0
        # for word, count in wordCounts.iteritems():
        for word, count in wordCounts.items():
            if count >= 3:
                featureIndexes[word] = index
                index += 1

        return featureIndexes
    else:
        features = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(cur_row, len(featureIndexes)),
                                 dtype=np.float64)
        if targets:
            return features, targets, item_ids
        else:
            return features, item_ids



def main():


    # dataTrain = pd.read_table("tsv/avito_train.tsv", nrows=300)
    # dataTest = pd.read_table("tsv/avito_test.tsv")
    # output_file = "avito_solution.csv"
    # pkl_out_file = "pkl/train_data.pkl"

    file_id = strftime("%d%H%M%S")

    rootFolder = "/home/vdimir/edu/ml/avito"

    trainFileName = "%s/avito_train/avito_train.tsv" % rootFolder
    tetsFileName = "%s/avito_train/avito_test.tsv" % rootFolder
    output_file = "%s/avito_solution/avito_solution%s.csv" % (rootFolder,file_id)
    pkl_out_file = "%s/avito_train_pkl/train_data.pkl" % rootFolder

    # trainSize = 300
    # dataTrain = pd.read_table(trainFileName, nrows=trainSize)
    # dataTest = pd.read_table(tetsFileName, nrows=trainSize)

    # dataTrainTest = pd.read_table(trainFileName, nrows=trainSize*2, header=None, skiprows=trainSize+1)
    # dataTrainTest.columns = dataTrain.columns

    # featureIndexes = processData(dataTrain)
    # trainFeatures, trainTargets, trainItemIds = processData(dataTrain, featureIndexes)
    # testFeatures, testItemIds = processData(dataTest, featureIndexes)
    # joblib.dump((trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds), pkl_out_file)

    trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(pkl_out_file)
    logging.info("Feature preparation done, fitting model...")
    swm = svm.LinearSVC(C=0.2)

    clf = CalibratedClassifierCV(swm)
    # clf = SGDClassifier(    loss="log",
    # penalty="l2",
    # alpha=1e-3,
    # class_weight="auto")
    clf.fit(trainFeatures, trainTargets)

    logging.info("Predicting...")

    predicted_scores = clf.predict_proba(testFeatures).T[1]

    logging.info("Write results...")
    logging.info("Writing submission to %s" % output_file)
    f = open(output_file, "w")
    f.write("id\n")

    for pred_score, item_id in sorted(zip(predicted_scores, testItemIds), reverse=True):
        f.write("%d\n" % (item_id))
    f.close()
    logging.info("Done.")


if __name__ == "__main__":
    main()


