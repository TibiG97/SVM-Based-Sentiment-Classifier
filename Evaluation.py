from InputProcessing import *
from NaiveBayes import *
from SVM import *
import subprocess
import sys


def get_platform():
    platforms = {
        'linux': 'Linux',
        'linux2': 'Linux',
        'darwin': 'OS X',
        'win32': 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform
    return platforms[sys.platform]


if get_platform() == 'Windows':
    learn_file = os.path.join(os.path.join(os.getcwd(), 'SVM-Light'), 'svm_learn_windows')
    classify_file = os.path.join(os.path.join(os.getcwd(), 'SVM-Light'), 'svm_classify_windows')
elif get_platform() == 'Linux':
    learn_file = os.path.join(os.path.join(os.getcwd(), 'SVM-Light'), 'svm_learn_linux')
    classify_file = os.path.join(os.path.join(os.getcwd(), 'SVM-Light'), 'svm_classify_linux')
elif get_platform() == 'OS X':
    learn_file = os.path.join(os.path.join(os.getcwd(), 'SVM-Light'), 'svm_learn_osx')
    classify_file = os.path.join(os.path.join(os.getcwd(), 'SVM-Light'), 'svm_classify_osx')

train_file = os.path.join(os.getcwd(), 'SVM-Train')
test_file = os.path.join(os.getcwd(), 'SVM-Test')
model_file = os.path.join(os.getcwd(), 'SVM-Model')
results_file = os.path.join(os.getcwd(), 'SVM-Classification')


def cross_validation(model, smoothed, features, mode, path_pos_reviews, path_neg_reviews, stemmed, no_of_splits, file):
    pos_reviews = create_reviews_list(path_pos_reviews, stemmed)
    neg_reviews = create_reviews_list(path_neg_reviews, stemmed)

    # printing the header of the evaluated Model
    print("Model: " + model, file=file)
    print("Smoothed: " + str(smoothed), file=file)
    print("Features: " + features, file=file)
    print("Mode: " + mode, file=file)
    print("Stemmed: " + str(stemmed), file=file)
    print("#Splits: " + str(no_of_splits), file=file)

    # initialize parameters for evaluation
    all_system_labels = {}
    accuracy = 0
    all_reviews = pos_reviews + neg_reviews
    splits = []
    splits_with_validation = stratified_split(all_reviews, no_of_splits)
    reviews_wo_validaton = []

    for iterator in range(0, len(splits_with_validation) - 1):
        for name in splits_with_validation[iterator]:
            reviews_wo_validaton.append(name)

    splits = stratified_split(reviews_wo_validaton, no_of_splits)

    original_labels = merge_dictionaries(label_reviews_list(pos_reviews, "positive"),
                                         label_reviews_list(neg_reviews, "negative"))

    (unigrams, unigrams_counter) = select_unigram_features(all_reviews, 5)
    (bigrams, bigrams_counter) = select_bigram_features(all_reviews, 10)

    # perform the n-fold cross validation
    for iterator in range(0, len(splits)):
        training_set = generate_training_set(splits, iterator)
        test_set = generate_test_set(splits, iterator)

        if model == "Naive Bayes":

            if features == "Unigrams":
                unigram_probs = generate_unigram_probs(training_set, original_labels, smoothed)
                unigram_feature_vectors = compute_unigram_feature_vectors(test_set, unigrams, True, mode)
                predicted_labels = naivebayes(unigram_probs, unigram_feature_vectors, test_set)

            elif features == "Bigrams":
                bigram_probs = generate_bigram_probs(training_set, original_labels, smoothed)
                bigram_feature_vectors = compute_bigram_feature_vectors(test_set, bigrams, True, mode)
                predicted_labels = naivebayes(bigram_probs, bigram_feature_vectors, test_set)

            elif features == "Both":
                both_probs = generate_both_probs(training_set, original_labels, smoothed)
                unigram_feature_vectors = compute_unigram_feature_vectors(test_set, unigrams, True, mode)
                bigram_feature_vectors = compute_bigram_feature_vectors(test_set, bigrams, True, mode)

                feature_vectors = {}
                for name in unigram_feature_vectors.keys():
                    feature_vectors[name] = merge_dictionaries(unigram_feature_vectors[name],
                                                               bigram_feature_vectors[name])

                predicted_labels = naivebayes(both_probs, feature_vectors, test_set)

        elif model == "SVM":
            predicted_labels = {}

            if features == "Unigrams":
                training_unigram_feature_vectors = compute_unigram_feature_vectors(training_set, unigrams, True, mode)
                test_unigram_feature_vectors = compute_unigram_feature_vectors(test_set, unigrams, True, mode)
                process_feature_vectors_to_svm_input(training_unigram_feature_vectors, unigrams_counter,
                                                     original_labels, "train")
                process_feature_vectors_to_svm_input(test_unigram_feature_vectors, unigrams_counter, original_labels,
                                                     "test")

            elif features == "Bigrams":
                training_bigram_feature_vectors = compute_bigram_feature_vectors(training_set, bigrams, True, mode)
                test_bigram_feature_vectors = compute_bigram_feature_vectors(test_set, bigrams, True, mode)
                process_feature_vectors_to_svm_input(training_bigram_feature_vectors, bigrams_counter, original_labels,
                                                     "train")
                process_feature_vectors_to_svm_input(test_bigram_feature_vectors, bigrams_counter, original_labels,
                                                     "test")

            elif features == "Both":
                training_unigram_feature_vectors = compute_unigram_feature_vectors(training_set, unigrams, True, mode)
                training_bigram_feature_vectors = compute_bigram_feature_vectors(training_set, bigrams, True, mode)
                training_feature_vectors = {}
                for name in training_unigram_feature_vectors.keys():
                    training_feature_vectors[name] = merge_dictionaries(training_unigram_feature_vectors[name],
                                                                        training_bigram_feature_vectors[name])

                test_unigram_feature_vectors = compute_unigram_feature_vectors(test_set, unigrams, True, mode)
                test_bigram_feature_vectors = compute_bigram_feature_vectors(test_set, bigrams, True, mode)
                test_feature_vectors = {}
                for name in test_unigram_feature_vectors.keys():
                    test_feature_vectors[name] = merge_dictionaries(test_unigram_feature_vectors[name],
                                                                    test_bigram_feature_vectors[name])

                for pair in bigrams_counter:
                    bigrams_counter[pair] += len(unigrams_counter)
                both_counter = merge_dictionaries(unigrams_counter, bigrams_counter)

                process_feature_vectors_to_svm_input(training_feature_vectors, both_counter, original_labels, "train")
                process_feature_vectors_to_svm_input(test_feature_vectors, both_counter, original_labels, "test")

            subprocess.call(learn_file + " -z c  " + train_file + " " + model_file, shell=True)
            subprocess.call(classify_file + " " + test_file + " " + model_file + " " + results_file, shell=True)

            classification_file = open("SVM-Classification", "r")
            for review in test_set:
                value = float(classification_file.readline())
                if value < 0:
                    predicted_labels[review.name] = "negative"
                else:
                    predicted_labels[review.name] = "positive"

        if iterator == 0:
            all_system_labels.update(predicted_labels.copy())

        score = 0
        for name in predicted_labels.keys():
            if predicted_labels[name] == original_labels[name]:
                score += 1
        split_accuracy = (score / len(test_set))
        accuracy += split_accuracy / len(splits)

    print("The average accuracy is: " + str(round(accuracy, 3)) + "\n", file=file)
    print(file=file)

    return all_system_labels
