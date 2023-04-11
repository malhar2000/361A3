import json
import sys
import os
import re
import csv
import math
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
IDF = "idf"
CENTROID = "centroid"


def readTrainFile(train_file):
    with open(train_file, 'r') as f:
        data = json.load(f)
    return data


def fileCheck(train_file, output_file):
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found")
        exit()
    if os.path.isfile(output_file):
        ans = input(
            f"Warning: {output_file} already exists. Would you like to overwrite (y/n)? ")
        if ans.lower() != 'y':
            exit()


def compute_centroid_vectors(data, idf_values, n):
    centroid_vectors = {}

    for doc in data:
        class_name = doc['category']
        each_doc_voc_counter = Counter()
        
        if class_name not in centroid_vectors:
            centroid_vectors[class_name] = [0] * len(idf_values)

        words = word_tokenize(doc['text'])
        for word in words:
            if word not in STOP_WORDS and word.isalpha():
                word = re.sub(r'[^\w\s]', '', word)
                each_doc_voc_counter.update([word])
        # Compute the vector for the document
        for word in each_doc_voc_counter:
            each_doc_voc_counter[word] = (
                (math.log(each_doc_voc_counter[word], 10) + 1) * math.log(n/idf_values[word], 10))
        norm = math.sqrt(
            sum(weight ** 2 for weight in each_doc_voc_counter.values()))
        for word in each_doc_voc_counter:
            centroid_vectors[class_name][list(idf_values.keys()).index(
                    word)] = each_doc_voc_counter[word] / norm
    return centroid_vectors


def main():

    if len(sys.argv) < 3:
        print("Usage: python3 rocchio_train.py <train.json> <model.tsv>")
        exit()

    fileCheck(sys.argv[1], sys.argv[2])

    train_data = readTrainFile(sys.argv[1])

    n = len(train_data)

    vocabulary = Counter()
    for doc in train_data:
        already_counted = set()
        words = word_tokenize(doc['text'])
        for word in words:
            if word not in STOP_WORDS and word.isalpha():
                word = re.sub(r'[^\w\s]', '', word)
                if word not in already_counted:
                    already_counted.add(word)
                    vocabulary.update([word])

    with open(sys.argv[2], 'w') as output_file:
        for word in vocabulary:
            csv.writer(output_file, delimiter='\t').writerow(
                [IDF, word, math.log(n/vocabulary[word], 10)])

        centroid_vectors = compute_centroid_vectors(train_data, vocabulary, n)
        for class_name in centroid_vectors:
                vector = centroid_vectors[class_name]
                output_file.write('centroid\t{}\t{}\n'.format(
                    class_name, ' '.join([str(x) for x in vector])))


if __name__ == "__main__":
    main()
