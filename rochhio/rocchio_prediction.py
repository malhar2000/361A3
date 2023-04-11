import json
import sys
import os
import re
import math
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

STOP_WORDS = set(nltk.corpus.stopwords.words('english'))


def fileCheck(model_file, test_file):
    if not os.path.exists(model_file):
        print(f"Error: {model_file} not found")
        exit()
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found")
        exit()


def readTestFile(test_file):
    with open(test_file, 'r') as f:
        data = json.load(f)
    return data


def readModelFile(model_file):
    with open(model_file, 'r') as f:
        idf = {}
        centroids = {}
        for line in f:
            line = line.strip().split('\t')
            if line[0] == 'idf':
                idf[line[1]] = float(line[2])
            elif line[0] == 'centroid':
                class_name = line[1]
                centroids[class_name] = [float(i) for i in line[2].split()]
    return idf, centroids


def preprocess_text(text):
    words = word_tokenize(text)
    doc = [re.sub(r'[^\w\s]', '', word)
                     for word in words if word not in STOP_WORDS and word.isalpha()]
    return Counter(doc)


def predict(vector, centroid_vectors):
    min_distance = float('inf')
    predicted_category = None
    for category, centroid in centroid_vectors.items():
        distance = 0
        for i in range(len(vector)):
            if vector[i] != 0:
                distance += (vector[i] - centroid[i])**2
        distance = math.sqrt(distance)
        if  distance < min_distance:
            min_distance = distance
            predicted_category = category
    return predicted_category


def evaluate_score(test_data, idf_values, centroid_vectors, categories):
    tp = {category: 0 for category in categories}
    tn = {category: 0 for category in categories}
    fp = {category: 0 for category in categories}
    fn = {category: 0 for category in categories}
    for doc in test_data:
        terms = preprocess_text(doc['text'])
        vector = [0] * len(idf_values)
        for term, count in terms.items():
            if term in idf_values:
                terms[term] = (math.log10(count) + 1) * idf_values[term]
        norm = math.sqrt(sum(weight ** 2 for weight in terms.values()))
        for term, tf_idf in terms.items():
            if term in idf_values: 
                score = tf_idf / norm
                vector[list(idf_values.keys()).index(term)] = score

        predicted_category = predict(vector, centroid_vectors)
        actual_category = doc['category']
        # print(f"Document {doc['category']} is classified as {predicted_category}")
        if predicted_category == actual_category:
            tp[actual_category] += 1
        else:
            fn[actual_category] += 1
            fp[predicted_category] += 1
        for category in categories:
            if category != actual_category and category != predicted_category:
                tn[category] += 1
    all_tp = sum(tp.values())
    all_fp = sum(fp.values())
    all_fn = sum(fn.values())
    micro_precision = all_tp / (all_tp + all_fp) if all_tp + all_fp != 0 else 0
    micro_recall = all_tp / (all_tp + all_fn) if all_tp + all_fn != 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / \
        (micro_precision + micro_recall) if micro_precision + \
        micro_recall != 0 else 0
    macro_f1 = sum([2 * tp[category] / (2 * tp[category] + fp[category] + fn[category]) if tp[category] +
                   fp[category] + fn[category] != 0 else 0 for category in categories]) / len(categories)

    for category in categories:
        p = tp[category] / (tp[category] + fp[category]
                            ) if tp[category] + fp[category] != 0 else 0
        r = tp[category] / (tp[category] + fn[category]
                            ) if tp[category] + fn[category] != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        print(
            f"{category} TP={tp[category]} FP={fp[category]} TN={tn[category]} FN={fn[category]} P={p:.4f} R={r:.4f} F1={f1:.4f}")

    print(f"all micro={micro_f1:.4f}")
    print(f"all macro={macro_f1:.4f}")



def main():
    if len(sys.argv) < 3:
        print("Usage: python3 rocchio_prediction.py <model_file> <test_data_file>")
        exit()

    fileCheck(sys.argv[1], sys.argv[2])
    data = readTestFile(sys.argv[2])

    idf, centroids = readModelFile(sys.argv[1])

    categories = list(centroids.keys())
    evaluate_score(data, idf, centroids, categories)


if __name__ == '__main__':
    main()
