import random
import os
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import collections
import nltk.metrics
from nltk.metrics.scores import (precision, recall)


# reads files, tokenizes, del punct, del stopwords, stems, makes feat table
def data_processing(verbose=False):
    labeled_reviews = []
    # Iterate through all files
    for c in ('pos', 'neg'):
        path = "./Homework2-Data/" + c
        for file in os.listdir(path):
            # Check whether file is in text format or not
            if file.endswith(".txt"):
                file_path = f"{path}/{file}"
                with open(file_path, 'r') as f:
                    labeled_reviews.append((f.read().lower(), c))       #[("I love this", 'pos'), ]
    # tokenize and remove punctuation
    tokens = set()
    for review in labeled_reviews:
        review_tokens = word_tokenize(review[0])
        for word in review_tokens:
            no_punc = ""
            for char in word:
                if char not in string.punctuation:
                    no_punc += char
            if no_punc:
                tokens.add(no_punc)
    # create feature table
    data = []
    stopword = stopwords.words("english")
    porter = PorterStemmer()
    i = 0
    # make feature table
    for x in labeled_reviews:
        i += 1
        if verbose:
            print(f'{i}/{len(labeled_reviews)}')                  # this takes like 7 min
        dict = {}
        for word in tokens:
            if word not in stopword:            # do not include stopwords
                is_in = word in x[0]
                stemmed = porter.stem(word)     # stemming
                if not dict.get(stemmed):       # don't want to ever change a True item to False
                    dict[stemmed] = is_in
        data.append((dict, x[1]))               # ({"go": False, "love": True...}. "pos")
    # data = [({word: (word in word_tokenize(x[0])) for word in tokens}, x[1]) for x in labeled_reviews]
    return data


# given data, returns all needed datasets for k-fold as list of (training set, testing set)
def tenfold_sets(data, k):
    random.shuffle(data)
    datasets = []
    for i in range(10):
        begin = (int)(len(data)*(i/k))
        end = (int)(len(data)*((i+1)/k))
        training = data[0:begin] + data[end:]
        testing = data[begin:end]
        datasets.append((training, testing))
    return datasets


def naive_bayes(datasets, k, verbose=False):
    pos_precisions = []
    neg_precisions = []
    pos_recalls = []
    neg_recalls = []
    # k-fold train and test
    for i in range(k):
        training = datasets[i][0]
        testing = datasets[i][1]
        classifier = nltk.NaiveBayesClassifier.train(training)
        # classifier.show_most_informative_features()

        # construct orig label and classifier dictionaries
        true_sets = collections.defaultdict(set)
        classifier_sets = collections.defaultdict(set)

        # run your classifier over testing dataset to see the performance
        for j, (doc, label) in enumerate(testing):
          true_sets[label].add(j)
          observed = classifier.classify(doc)
          classifier_sets[observed].add(j)

        pos_precisions.append(precision(true_sets['pos'], classifier_sets['pos']))
        neg_precisions.append(precision(true_sets["neg"], classifier_sets["neg"]))
        pos_recalls.append(recall(true_sets['pos'], classifier_sets['pos']))
        neg_recalls.append(recall(true_sets["neg"], classifier_sets["neg"]))
        if verbose:
            print('\nk =', i+1)
            print('pos_precision:', pos_precisions[-1])
            print('neg_precision:', neg_precisions[-1])
            print('pos_recall:', pos_recalls[-1])
            print('neg_recall:', neg_recalls[-1])

    print('\naverage pos precision:', sum(pos_precisions)/len(pos_precisions))
    print('average neg precision:', sum(neg_precisions)/len(neg_precisions))
    print('\naverage pos recall:', sum(pos_recalls)/len(pos_recalls))
    print('average neg recalls:', sum(neg_recalls)/len(neg_recalls))


def main():
    k = 10
    datasets = tenfold_sets(data_processing(True), k)
    naive_bayes(datasets, k, True)


if __name__ == "__main__":
    main()