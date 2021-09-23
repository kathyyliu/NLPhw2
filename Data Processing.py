import random
import os
from nltk.tokenize import word_tokenize
import collections
import nltk.metrics
from nltk.metrics.scores import (precision, recall)


# data processing
# I want to pair each data point to its class
#[(d, class), (d2, class)]
#[("I love this", 'pos'), ]

labeled_reviews = []
# iterate through all file
for c in ('pos', 'neg'):
    path = "./Homework2-Data/" + c
    for file in os.listdir(path):
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path}/{file}"
            with open(file_path, 'r') as f:
                labeled_reviews.append((f.read().lower(), c))
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

random.shuffle(data)                    # randomize the order
training = data[0:(int)(len(data)/2)]   # split into training and testing
testing = data[(int)(len(data)/2):]

classifier = nltk.NaiveBayesClassifier.train(training)
# most informative features found in the classifier
classifier.show_most_informative_features()

# construct two dictionaries
# one for the original label
# one for the classifier label
# calculate the % of TP and Fp

truesets = collections.defaultdict(set)
classifiersets =  collections.defaultdict(set)
# you want to look at precision and recall in both training anf testing
# if your performnace is really good in training but horrible in testing
# that means your model is overfitted
#
# for i, (doc, label) in enumerate(testing):
#   #run your classifier over testing dataset to see the peromance
#   truesets[label].add(i)
#   observed = classifier.classify(doc)
#   classifiersets[observed].add(i)
#
# print(truesets)
# print(classifiersets)
# # [1, 3, 4, 5, 19, 45]
# # [2, 3, 4, 19, 25, 40]
#
# pos_precision = precision(truesets['pos'], classifiersets['pos'])
# neg_precision = precision(truesets["neg"], classifiersets["neg"])
# print(pos_precision)
# print(neg_precision)
