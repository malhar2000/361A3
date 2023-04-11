import sys
import json
import os
from collections import defaultdict
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


def extract_vocabulary(training_data):
    """
    This function extracts vocabulary from the training data and excludes stopwords to remove noise and focus
    on more informative words.

    Parameters:
    training_data (list): Contains a list of dictionaries representing the training data, where each dictionary
    contains a "text" key with the text of a document as its value.

    Returns:
    vocabulary (set): A set of unique words in the training data after excluding English stopwords. If the training
    data is empty or contains no words, an empty set is returned.
    """
    vocabulary = set()
    english_stopwords = set(stopwords.words('english'))

    for entry in training_data:
        words = entry["text"].split()
        vocabulary.update(words)

    # Remove stopwords from the vocabulary
    vocabulary = vocabulary.difference(english_stopwords)

    return vocabulary


def train_multinomial_nb(classes, training_data):
    """
    This function trains a Multinomial Naive Bayes Classifier using the given training data and class labels.
    It computes the prior probabilities and conditional probabilities for each term in the vocabulary given the class.

    Parameters:
    classes (list): A list of class labels (categories) present in the training data.
    training_data (list): A list of dictionaries containing the text and category for each document.

    Returns:
    vocabulary (set): The set of unique words in the training data excluding English stopwords.
    priors (dict): A dictionary containing the prior probabilities of each class.
    cond_probs (dict): A nested dictionary containing the conditional probabilities of each term given a class.
    """
    vocabulary = extract_vocabulary(training_data)
    total_docs = len(training_data)
    priors = {}
    cond_probs = defaultdict(dict)

    class_term_counts = defaultdict(lambda: defaultdict(int))
    class_total_tokens = defaultdict(int)
    class_docs = {class_name: 0 for class_name in classes}

    for entry in training_data:
        class_name = entry["category"]
        class_docs[class_name] += 1
        for word in entry["text"].split():
            if word in vocabulary:
                class_term_counts[class_name][word] += 1
                class_total_tokens[class_name] += 1

    for class_name in classes:
        priors[class_name] = class_docs[class_name] / total_docs
        total_tokens = class_total_tokens[class_name]

        for term in vocabulary:
            term_count = class_term_counts[class_name][term]
            cond_probs[term][class_name] = (term_count + 1) / (total_tokens + len(vocabulary))

    return vocabulary, priors, cond_probs


def write_model(model_path, priors, cond_probs):
    """
    This function writes the trained Multinomial Naive Bayes Classifier model to a TSV (Tab-separated values) file.
    The model file includes the prior probabilities for each class and the conditional probabilities for each term
    given a class.

    Parameters:
    model_path (str): The path to the file where the model will be saved.
    priors (dict): A dictionary containing the prior probabilities of each class.
    cond_probs (dict): A nested dictionary containing the conditional probabilities of each term given a class.

    Returns:
    None
    """
    with open(model_path, "w") as f:
        for class_name, prior in priors.items():
            f.write(f"prior\t{class_name}\t{prior}\n")
        for term, class_probs in cond_probs.items():
            for class_name, prob in class_probs.items():
                f.write(f"likelihood\t{class_name}\t{term}\t{prob}\n")


def main():
    """
    The main function is the entry point of the script. It reads the training data file and trains a Multinomial
    Naive Bayes Classifier.
    The model is then saved to a TSV file. If the specified output file already exists, the user will be asked for
    confirmation to overwrite it.
    """
    train_data_path = sys.argv[1]
    model_path = sys.argv[2]

    if os.path.exists(model_path):
        confirm = input(f"File '{model_path}' already exists. Do you want to overwrite it? [y/N]: ")
        if not confirm.lower().startswith('y'):
            print("Aborting.")
            sys.exit(0)

    with open(train_data_path, "r") as f:
        training_data = json.load(f)

    classes = ["business", "entertainment", "politics", "sport", "tech"]

    print("Start training...")
    vocabulary, priors, cond_probs = train_multinomial_nb(classes, training_data)
    print("Training completed.")

    print("Writing model to TSV file...")
    write_model(model_path, priors, cond_probs)
    print(f"Model saved to {model_path}.")


if __name__ == "__main__":
    main()
