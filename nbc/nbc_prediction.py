import sys
import json
import math
from collections import defaultdict, Counter


def retrieve_model(model_path):
    """
    This function reads a trained Multinomial Naive Bayes model from a TSV file and returns the priors and
    conditional probabilities.

    Parameters:
    model_path (str): The path to the TSV file containing the trained model.

    Returns:
    priors (dict): A dictionary containing the prior probabilities for each class.
    cond_probs (defaultdict): A nested defaultdict containing the conditional probabilities for each term in each class.
    """
    priors = {}
    cond_probs = defaultdict(dict)

    with open(model_path, "r") as f:
        for line in f:
            tokens = line.strip().split("\t")
            if tokens[0] == "prior":
                priors[tokens[1]] = float(tokens[2])
            elif tokens[0] == "likelihood":
                cond_probs[tokens[2]][tokens[1]] = float(tokens[3])

    return priors, cond_probs


def apply_multinomial_nb(classes, vocabulary, priors, cond_probs, doc):
    """
    This function applies the multinomial Naive Bayes algorithm to classify a document based on its content.

    Parameters:
    classes (list): A list of class names to consider for classification.
    vocabulary (set): A set of unique words representing the vocabulary.
    priors (dict): A dictionary containing the prior probabilities for each class.
    cond_probs (dict): A dictionary containing the conditional probabilities for each term and class.
    doc (str): A string representing the document to be classified.

    Returns:
    the best class (str): The class with the highest score according to the multinomial Naive Bayes algorithm.
    """
    words = doc.split()
    tokens = []
    for word in words:
        if word in vocabulary:
            tokens.append(word)
    scores = {}

    for class_name in classes:
        score = math.log(priors[class_name])
        for token in tokens:
            score += math.log(cond_probs[token][class_name])
        scores[class_name] = score

    return max(scores, key=scores.get)


def calculate_metrics(confusion_matrix):
    """
    This function calculates performance metrics (precision, recall, F1-score) for a confusion matrix.

    Parameters:
    confusion_matrix (dict): A dictionary containing the confusion matrix values (true positive, false positive,
    false negative, true negative) for each class.

    Returns:
    precision (dict): A dictionary containing the precision values for each class.
    recall (dict): A dictionary containing the recall values for each class.
    f1 (dict): A dictionary containing the F1-score values for each class.
    micro_f1 (float): The micro-averaged F1-score across all classes.
    macro_f1 (float): The macro-averaged F1-score across all classes.
    """

    precision = {}
    recall = {}
    f1 = {}

    for class_name, cm_values in confusion_matrix.items():
        tp = cm_values["tp"]
        fp = cm_values["fp"]
        fn = cm_values["fn"]
        tn = cm_values["tn"]

        precision[class_name] = tp / (tp + fp) if tp + fp > 0 else 0
        recall[class_name] = tp / (tp + fn) if tp + fn > 0 else 0
        f1[class_name] = 2 * (precision[class_name] * recall[class_name]) / (
                precision[class_name] + recall[class_name]) if precision[class_name] + recall[class_name] > 0 else 0

    micro_f1 = sum([cm_values["tp"] for cm_values in confusion_matrix.values()]) / (
            sum([cm_values["tp"] + cm_values["fp"] for cm_values in confusion_matrix.values()]) or 1)
    macro_f1 = sum(f1.values()) / len(f1)

    return precision, recall, f1, micro_f1, macro_f1


def main():
    """
    This function evaluates the performance of the trained Multinomial Naive Bayes model on a test dataset.
    It calculates and prints the performance metrics for each class.
    The command-line arguments are used to specify the paths to the model and test data files. The function processes
    the test data, uses the Multinomial Naive Bayes model, and calculates the confusion matrix, which is then used to
    compute the performance metrics.
    """
    model_path = sys.argv[1]
    test_data_path = sys.argv[2]

    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    priors, cond_probs = retrieve_model(model_path)
    classes = list(priors.keys())
    vocabulary = set(cond_probs.keys())

    confusion_matrix = {class_name: Counter({"tp": 0, "fp": 0, "fn": 0, "tn": 0}) for class_name in classes}

    for entry in test_data:
        true_class = entry["category"]
        predicted_class = apply_multinomial_nb(classes, vocabulary, priors, cond_probs, entry["text"])

        for class_name in classes:
            if class_name == true_class:
                if class_name == predicted_class:
                    confusion_matrix[class_name]["tp"] += 1
                else:
                    confusion_matrix[class_name]["fn"] += 1
            else:
                if class_name == predicted_class:
                    confusion_matrix[class_name]["fp"] += 1
                else:
                    confusion_matrix[class_name]["tn"] += 1

    precision, recall, f1, micro_f1, macro_f1 = calculate_metrics(confusion_matrix)

    for class_name in classes:
        print(
            f"{class_name}: TP={confusion_matrix[class_name]['tp']} FP={confusion_matrix[class_name]['fp']} "
            f"FN={confusion_matrix[class_name]['fn']} TN={confusion_matrix[class_name]['tn']} "
            f"Precision={precision[class_name]:.4f} Recall={recall[class_name]:.4f} F1={f1[class_name]:.4f}")

    print(f"Micro-averaged F1: {micro_f1:.4f}")
    print(f"Macro-averaged F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
