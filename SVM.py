def select_unigram_features(reviews, cutoff):
    word_count = {}
    unigram_vector = []
    unigram_position = {}

    for review in reviews:
        for word in review.content:
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1

    for word in word_count.keys():
        if word_count[word] >= cutoff:
            unigram_vector.append(word)

    counter = 1
    for word in unigram_vector:
        unigram_position[word] = counter
        counter += 1

    return unigram_vector, unigram_position


def select_bigram_features(reviews, cutoff):
    pair_count = {}
    bigram_vector = []
    bigram_positions = {}

    for review in reviews:
        for this_word, next_word in zip(review.content[:-1], review.content[1:]):
            if (this_word, next_word) not in pair_count:
                pair_count[(this_word, next_word)] = 1
            else:
                pair_count[(this_word, next_word)] += 1

    for pair in pair_count.keys():
        if pair_count[pair] >= cutoff:
            bigram_vector.append(pair)

    counter = 1
    for pair in bigram_vector:
        bigram_positions[pair] = counter
        counter += 1

    return bigram_vector, bigram_positions


def compute_unigram_feature_vectors(reviews, unigram_features, non_zero, mode):
    unigram_feature_vectors = {}

    for review in reviews:
        feature_vector = {}

        for feature in unigram_features:
            feature_vector[feature] = 0

        for word in review.content:
            if word in feature_vector:
                feature_vector[word] += 1

        if mode == "Presence":
            for word in feature_vector:
                if feature_vector[word] > 1:
                    feature_vector[word] = 1

        if non_zero:
            unigram_feature_vectors[review.name] = {x: y for x, y in feature_vector.items() if y != 0}
        else:
            unigram_feature_vectors[review.name] = feature_vector

    return unigram_feature_vectors


def compute_bigram_feature_vectors(reviews, bigram_features, non_zero, mode):
    bigram_feature_vectors = {}

    for review in reviews:
        feature_vector = {}

        for feature in bigram_features:
            feature_vector[feature] = 0

        for this_word, next_word in zip(review.content[:-1], review.content[1:]):
            if (this_word, next_word) in feature_vector:
                feature_vector[(this_word, next_word)] += 1

        if mode == "Presence":
            for pair in feature_vector:
                if feature_vector[pair] > 1:
                    feature_vector[pair] = 1

        if non_zero:
            bigram_feature_vectors[review.name] = {x: y for x, y in feature_vector.items() if y != 0}
        else:
            bigram_feature_vectors[review.name] = feature_vector

    return bigram_feature_vectors


def process_feature_vectors_to_svm_input(feature_vectors, positions, original_labels, file_type):
    if file_type == "train":
        file = open("SVM-Train", "a")
    elif file_type == "test":
        file = open("SVM-Test", "a")
    file.truncate(0)

    for name in feature_vectors.keys():
        string_representation = ""

        if original_labels[name] == "positive":
            string_representation += "1 "
        else:
            string_representation += "-1 "

        for feature in feature_vectors[name].keys():
            string_representation += str(positions[feature]) + ":" + str(feature_vectors[name][feature]) + " "

        print(string_representation, file=file)

    return
