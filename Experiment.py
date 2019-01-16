from timeit import default_timer as timer
from Evaluation import cross_validation
from SignTest import compute_p_value
from InputProcessing import *

path_pos_reviews = os.path.join(os.getcwd(), 'POS')
path_neg_reviews = os.path.join(os.getcwd(), 'NEG')
no_of_splits = 10

gram = ["Unigrams", "Bigrams", "Both"]
mode = ["Frequency", "Presence"]
model = ["Naive Bayes", "SVM"]
stemmed = [False, True]
smoothed = [False, True]

nb_labels = []
svm_labels = []

nb_results_file = open("NB-Results", "a")
nb_results_file.truncate(0)

svm_results_file = open("SVM-Results", "a")
svm_results_file.truncate(0)

sign_test_results_file = open("SignTest-Results", "a")
sign_test_results_file.truncate(0)

# run cross-validation for NB for all cases
for mymode in mode:
    for mygram in gram:
        start = timer()
        nb_labels.append(cross_validation(model[0], smoothed[1], mygram, mymode, path_pos_reviews,
                                          path_neg_reviews, stemmed[0], no_of_splits, nb_results_file))
        stop = timer()
        print(stop - start)

# run cross-validation for SVM for all cases
for mymode in mode:
    for mygram in gram:
        start = timer()
        svm_labels.append(cross_validation(model[1], smoothed[1], mygram, mymode, path_pos_reviews,
                                           path_neg_reviews, stemmed[0], no_of_splits, svm_results_file))
        stop = timer()
        print(stop - start)

all_system_labels = {}
pos_reviews = create_reviews_list(path_pos_reviews, stemmed[0])
neg_reviews = create_reviews_list(path_neg_reviews, stemmed[0])
original_labels = merge_dictionaries(label_reviews_list(pos_reviews, "positive"),
                                     label_reviews_list(neg_reviews, "negative"))

# compute p_values
print("p(SVM_frequency_unigrams, SVM_presence_unigrams) = " +
      str(round(compute_p_value(original_labels, svm_labels[0], svm_labels[3]), 3)),
      file=sign_test_results_file)
print(file=sign_test_results_file)

print("p(SVM_frequency_bigrams, SVM_presence_bigrams) = " +
      str(round(compute_p_value(original_labels, svm_labels[1], svm_labels[4]), 3)),
      file=sign_test_results_file)
print(file=sign_test_results_file)

print("p(SVM_frequency_both, SVM_presence_both) = " +
      str(round(compute_p_value(original_labels, svm_labels[2], svm_labels[5]), 3)),
      file=sign_test_results_file)
print(file=sign_test_results_file)

print("p(NB_presence_unigrams, SVM_presence_unigrams) = " +
      str(round(compute_p_value(original_labels, nb_labels[3], svm_labels[3]), 3)),
      file=sign_test_results_file)
print(file=sign_test_results_file)

print("p(NB_presence_bigrams, SVM_presence_bigrams) = " +
      str(round(compute_p_value(original_labels, nb_labels[4], svm_labels[4]), 3)),
      file=sign_test_results_file)
print(file=sign_test_results_file)

print("p(NB_presence_both, SVM_presence_both) = " +
      str(round(compute_p_value(original_labels, nb_labels[5], svm_labels[5]), 3)),
      file=sign_test_results_file)
print(file=sign_test_results_file)
