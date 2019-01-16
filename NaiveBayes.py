import math


def generate_unigram_probs(training_set, original_labels, smoothed):
    probs = {}
    word_set = set()
    pos_count = {}
    neg_count = {}
    pos_total = 0
    neg_total = 0

    for review in training_set:
        for word in review.content:
            word_set.add(word)

    for word in word_set:
        pos_count[word] = 0
        neg_count[word] = 0

    for review in training_set:
        for word in review.content:
            if original_labels[review.name] == "positive":
                pos_count[word] += 1
            else:
                neg_count[word] += 1

    for word in word_set:
        pos_total += pos_count[word]
        neg_total += neg_count[word]

    if smoothed:
        for word in word_set:
            pos_prob = (pos_count[word] + 1) / (pos_total + len(word_set))
            neg_prob = (neg_count[word] + 1) / (neg_total + len(word_set))
            probs[word] = (pos_prob, neg_prob)
    else:
        for word in word_set:
            pos_prob = pos_count[word] / pos_total
            neg_prob = neg_count[word] / neg_total
            probs[word] = (pos_prob, neg_prob)

    return probs


def generate_bigram_probs(training_set, original_labels, smoothed):
    probs = {}
    pair_set = set()
    pos_count = {}
    neg_count = {}
    pos_total = 0
    neg_total = 0

    for review in training_set:
        for this_word, next_word in zip(review.content[:-1], review.content[1:]):
            pair_set.add((this_word, next_word))

    for pair in pair_set:
        pos_count[pair] = 0
        neg_count[pair] = 0

    for review in training_set:
        for this_word, next_word in zip(review.content[:-1], review.content[1:]):
            if original_labels[review.name] == "positive":
                pos_count[(this_word, next_word)] += 1
            else:
                neg_count[(this_word, next_word)] += 1

    for pair in pair_set:
        pos_total += pos_count[pair]
        neg_total += neg_count[pair]

    if smoothed:
        for pair in pair_set:
            pos_prob = (pos_count[pair] + 1) / (pos_total + len(pair_set))
            neg_prob = (neg_count[pair] + 1) / (neg_total + len(pair_set))
            probs[pair] = (pos_prob, neg_prob)
    else:
        for pair in pair_set:
            pos_prob = pos_count[pair] / pos_total
            neg_prob = neg_count[pair] / neg_total
            probs[pair] = (pos_prob, neg_prob)

    return probs


def generate_both_probs(training_set, original_labels, smoothed):
    probs = {}
    word_set = set()
    pos_count = {}
    neg_count = {}
    pos_total = 0
    neg_total = 0

    for review in training_set:
        for word in review.content:
            word_set.add(word)

    for word in word_set:
        pos_count[word] = 0
        neg_count[word] = 0

    for review in training_set:
        for word in review.content:
            if original_labels[review.name] == "positive":
                pos_count[word] += 1
            else:
                neg_count[word] += 1

    for word in word_set:
        pos_total += pos_count[word]
        neg_total += neg_count[word]

    if smoothed:
        for word in word_set:
            pos_prob = (pos_count[word] + 1) / (pos_total + len(word_set))
            neg_prob = (neg_count[word] + 1) / (neg_total + len(word_set))
            probs[word] = (pos_prob, neg_prob)
    else:
        for word in word_set:
            pos_prob = pos_count[word] / pos_total
            neg_prob = neg_count[word] / neg_total
            probs[word] = (pos_prob, neg_prob)

    pair_set = set()
    pos_count = {}
    neg_count = {}
    pos_total = 0
    neg_total = 0

    for review in training_set:
        for this_word, next_word in zip(review.content[:-1], review.content[1:]):
            pair_set.add((this_word, next_word))

    for pair in pair_set:
        pos_count[pair] = 0
        neg_count[pair] = 0

    for review in training_set:
        for this_word, next_word in zip(review.content[:-1], review.content[1:]):
            if original_labels[review.name] == "positive":
                pos_count[(this_word, next_word)] += 1
            else:
                neg_count[(this_word, next_word)] += 1

    for pair in pair_set:
        pos_total += pos_count[pair]
        neg_total += neg_count[pair]

    if smoothed:
        for pair in pair_set:
            pos_prob = (pos_count[pair] + 1) / (pos_total + len(pair_set))
            neg_prob = (neg_count[pair] + 1) / (neg_total + len(pair_set))
            probs[pair] = (pos_prob, neg_prob)
    else:
        for pair in pair_set:
            pos_prob = pos_count[pair] / pos_total
            neg_prob = neg_count[pair] / neg_total
            probs[pair] = (pos_prob, neg_prob)

    return probs


def naivebayes(probabilities, feature_vectors, test_set):
    predicted_labels = {}

    for review in test_set:
        pos_sum = 0
        neg_sum = 0

        for feature in feature_vectors[review.name].keys():
            if feature in probabilities.keys():
                if probabilities[feature][0] != 0:
                    pos_sum += math.log(probabilities[feature][0], 2) * feature_vectors[review.name][feature]
                if probabilities[feature][1] != 0:
                    neg_sum += math.log(probabilities[feature][1], 2) * feature_vectors[review.name][feature]

        if pos_sum > neg_sum:
            predicted_labels[review.name] = "positive"
        else:
            predicted_labels[review.name] = "negative"

    return predicted_labels
