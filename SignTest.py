import math


def ncr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n - r)


def compute_p_value(original_labels, model1_labels, model2_labels):
    null = 0
    minus = 0
    plus = 0
    p_value = 0

    for name in model1_labels.keys():
        if model1_labels[name] == model2_labels[name]:
            null += 1
        elif model1_labels[name] == original_labels[name]:
            plus += 1
        elif model2_labels[name] == original_labels[name]:
            minus += 1

    n = 2 * math.ceil(null / 2) + plus + minus
    k = math.ceil(null / 2) + min(plus, minus)

    for iterator in range(0, k + 1):
        p_value += ncr(n, iterator)
    p_value = p_value / pow(2, n - 1)

    return p_value
