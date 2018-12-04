import numpy as np
import heapq
import sys


def dist(u, v):
    return np.dot(u, v) / (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v)))


def most_similar(word, k):
    distances = [(curr_word, dist(word_to_vec[word], word_to_vec[curr_word])) for curr_word in words]
    return heapq.nlargest(k, distances, key=lambda tup: tup[1])


if __name__ == "__main__":
    vecs = np.loadtxt('wordVectors.txt')
    with open('vocab.txt', 'r') as f:
        words = f.readlines()
        words = [word.strip() for word in words]
        word_to_vec = {word: vec for word, vec in zip(words, vecs)}
    print(most_similar(sys.argv[1], 6))
