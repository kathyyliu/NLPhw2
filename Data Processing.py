import random
import os
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import collections
import nltk.metrics
from nltk.metrics.scores import (precision, recall)
from nltk.classify import MaxentClassifier


# reads files, tokenizes, del punct, del stopwords, stems, makes feat table
def data_processing(verbose=False):
    labeled_reviews = []
    nums = {'pos':[], 'neg':[]}
    # Iterate through all review files
    for c in ('pos', 'neg'):
        path = "./Homework2-Data/" + c
        i = 0
        for file in os.listdir(path):
            if file.endswith(".txt"):
                nums[c].append(int(file[file.index('_') + 1:file.index('.')]))          # get file number to match to rating
                file_path = f"{path}/{file}"
                with open(file_path, 'r') as f:
                    labeled_reviews.append((f.read().lower(), c))       #[("I love this", 'pos'), ]
                i += 1
            if i >= 100:
                break
    ordered_ratings = order_ratings(nums['pos'], nums['neg'])
    # tokenize and remove punctuation
    stopword = stopwords.words("english")
    tokens = set()
    for review in labeled_reviews:
        review_tokens = word_tokenize(review[0])
        for word in review_tokens:
            no_punc = ""
            for char in word:
                if char not in string.punctuation:
                    no_punc += char
            if no_punc and no_punc not in stopword:
                tokens.add(no_punc)
    # create feature table
    data = []
    porter = PorterStemmer()
    i = 0
    # make feature table
    for x in labeled_reviews:
        i += 1
        if verbose:
            print(f'{i}/{len(labeled_reviews)}')
        dict = {}
        for word in tokens:
            is_in = word in x[0]
            stemmed = porter.stem(word)     # stemming
            if not dict.get(stemmed):       # don't want to ever change a True item to False
                dict[stemmed] = is_in
        data.append((dict, x[1]))               # ({"go": False, "love": True...}. "pos")
    # data = [({word: (word in word_tokenize(x[0])) for word in tokens}, x[1]) for x in labeled_reviews]
    return data, ordered_ratings


# returns dict of list of ratings for pos, neg, according to given order of reviews
def order_ratings(pos_order, neg_order):
    ordered_ratings = {'pos':[], 'neg':[]}
    for c in ('positive', 'negative'):
        path = './Homework2-Data/ratings/' + c + '.txt'
        c_ratings = []
        with open(path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                rating = line[line.index(' ')+1:].strip()    # slice for only rating
                c_ratings.append(rating)
        c_abrv = c[:3]
        order = pos_order if c_abrv == 'pos' else neg_order
        # grabs only the ratings according to file number in the given order
        for num in order:
            ordered_ratings[c_abrv].append(c_ratings[num-1])
    return ordered_ratings


# given data, returns all needed datasets for k-fold as list of (training set, testing set)
def tenfold_sets(data, k):
    shuffled = data.copy()
    random.shuffle(shuffled)
    datasets = []
    for i in range(k):
        begin = (int)(len(shuffled) * (i / k))
        end = (int)(len(shuffled) * ((i + 1) / k))
        training = shuffled[0:begin] + shuffled[end:]
        testing = shuffled[begin:end]
        datasets.append((training, testing))
    return datasets


def naive_bayes(datasets, k, verbose=False):
    pos_precisions = []
    neg_precisions = []
    pos_recalls = []
    neg_recalls = []
    print('\n######### Naive Bayes #########')
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

    return classifier


def logistic_regression(datasets, k, verbose=False):
    pos_precisions = []
    neg_precisions = []
    pos_recalls = []
    neg_recalls = []
    print('\n######### Logistic Regression #########\n')
    # k-fold train and test
    for i in range(k):
        training = datasets[i][0]
        testing = datasets[i][1]
        classifier = MaxentClassifier.train(training, algorithm='GIS', max_iter=10)
        # classifier.show_most_informative_features(10)

        # construct orig label and classifier dictionaries
        true_sets = collections.defaultdict(set)
        classifier_sets = collections.defaultdict(set)

        # run your classifier over testing dataset to see the performance
        for j, (doc, label) in enumerate(testing):
            true_sets[label].add(j)
            observed = classifier.classify(doc)
            classifier_sets[observed].add(j)

        for c in ('pos', 'neg'):
            if not classifier_sets[c]:
                classifier_sets[c] = set()

        pos_precisions.append(precision(true_sets['pos'], classifier_sets['pos']))
        neg_precisions.append(precision(true_sets["neg"], classifier_sets["neg"]))
        pos_recalls.append(recall(true_sets['pos'], classifier_sets['pos']))
        neg_recalls.append(recall(true_sets["neg"], classifier_sets["neg"]))
        if verbose:
            print('\nk =', i + 1)
            print('pos_precision:', pos_precisions[-1])
            print('neg_precision:', neg_precisions[-1])
            print('pos_recall:', pos_recalls[-1])
            print('neg_recall:', neg_recalls[-1])

    print('\naverage pos precision:', sum(pos_precisions) / len(pos_precisions))
    print('average neg precision:', sum(neg_precisions) / len(neg_precisions))
    print('\naverage pos recall:', sum(pos_recalls) / len(pos_recalls))
    print('average neg recalls:', sum(neg_recalls) / len(neg_recalls))


def find_fake_reviews(model, testing, rating):

    fake = []
    for i in range(len(testing)):
        observed = model.classify(testing[i][0])
        # review fake if rating is 1 or 2 and prediction is pos, or 4,5 and neg
        if (float(rating[i]) < 3 and observed == 'pos') or (float(rating[i]) > 3 and observed == 'neg'):
            fake.append(testing[i])
    return fake


def main():
    k = 4
    data, ratings = data_processing(True)
    datasets = tenfold_sets(data, k)
    model = naive_bayes(datasets, k, True)
    logistic_regression(datasets, k, True)

    # shuffle data and respective ratings in same order
    temp = list(zip(data, ratings['pos'] + ratings['neg']))
    random.shuffle(temp)
    data, ratings = zip(*temp)
    print('\nfake reviews:\n', find_fake_reviews(model, data[:200], ratings[:200]))


if __name__ == "__main__":
    main()