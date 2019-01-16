import os
from stemming.porter2 import stem


class Review:
    def __init__(self, name, content):
        self.name = name
        self.content = content


def merge_dictionaries(dict_a, dict_b):
    merge = dict_a.copy()
    merge.update(dict_b)
    return merge


def create_reviews_list(path, stemmed):
    reviews = []
    file_names = os.listdir(path)

    for name in file_names:
        with open(path + "/" + name, "r", encoding="utf-8") as file:
            content = []
            for line in file:
                if stemmed:
                    content.append(stem(line))
                else:
                    content.append(line)
            reviews.append(Review(name, content))

    return reviews


def label_reviews_list(reviews, label):
    labels = {}
    for review in reviews:
        labels[review.name] = label
    return labels


def stratified_split(reviews, strats):
    splits = [[] for _ in range(strats)]
    counter = 0
    for review in reviews:
        splits[counter % strats].append(review)
        counter += 1
    return splits


def generate_training_set(splits, index):
    training_set = []
    for iterator in range(0, len(splits)):
        if iterator != index:
            training_set += splits[iterator]
    return training_set


def generate_test_set(splits, index):
    test_set = splits[index]
    return test_set
