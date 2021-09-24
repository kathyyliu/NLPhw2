import random
import os
from nltk.tokenize import word_tokenize
import collections
import nltk.metrics
from nltk.metrics.scores import (precision, recall)


# data processing
labeled_reviews = []
# iterate through all file
for c in ('pos', 'neg'):
    path = "./Homework2-Data/" + c
    for file in os.listdir(path):
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path}/{file}"
            with open(file_path, 'r') as f:
                labeled_reviews.append((f.read().lower(), c))       #[("I love this", 'pos'), ]
print(labeled_reviews)

tokens = set(word for words in labeled_reviews for word in word_tokenize(words[0]))
print(tokens)

data = []
# make feature table
for x in labeled_reviews:
    dict = {}
    for word in tokens:
        dict[word] = (word in x[0])
    data.append((dict, x[1]))   # ({"go": False, "love": True...}. "pos")
# data = [({word: (word in word_tokenize(x[0])) for word in tokens}, x[1]) for x in labeled_reviews]

random.shuffle(data)
# 10-fold cross validation
for i in range(10):
    print()
    begin = (int)(len(data)*(i/10))
    end = (int)(len(data)*((i+1)/10))
    testing = data[begin:end]
    training = data[0:begin] + data[end:]

    classifier = nltk.NaiveBayesClassifier.train(training)
    # classifier.show_most_informative_features()

    # construct orig label and classifier dictionaries
    truesets = collections.defaultdict(set)
    classifiersets = collections.defaultdict(set)

    # run your classifier over testing dataset to see the performance
    for j, (doc, label) in enumerate(testing):
      truesets[label].add(j)
      observed = classifier.classify(doc)
      classifiersets[observed].add(j)

    pos_precision = precision(truesets['pos'], classifiersets['pos'])
    neg_precision = precision(truesets["neg"], classifiersets["neg"])
    print('k =', i+1)
    print('pos_precision:', pos_precision)
    print('neg_precision:', neg_precision)

    pos_recall = recall(truesets['pos'], classifiersets['pos'])
    neg_recall = recall(truesets["neg"], classifiersets["neg"])
    print('pos_recall:', pos_recall)
    print('neg_recall:', neg_recall)

