import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import nltk

CURRENT_DIR = os.getcwd()
DATA_DIR = ""
if "src" in CURRENT_DIR:
    DATA_DIR = CURRENT_DIR.replace("src", "data")
elif CURRENT_DIR.split('/')[-1] == "cs175-project":
    DATA_DIR = CURRENT_DIR + "/data"

# print("Loading word frequencies from articles..")
# art_freq = pickle.load(open(DATA_DIR + "/all_articles_frequencies", "rb"))
try:
    art_freq = pickle.load(open(DATA_DIR + "/fake_articles_frequencies", "rb"))
except:
    fake_articles = os.listdir(DATA_DIR + "/generated_articles/")
    data_set = []
    print("Loading fake articles")
    for fa in fake_articles:
        if fa != '.DS_Store':
            f = open(DATA_DIR + "/generated_articles/" + fa)
            data_set.extend(nltk.word_tokenize(f.readline()))

    art_freq = nltk.FreqDist(data_set)
    pickle.dump(art_freq, open(DATA_DIR + "/fake_articles_frequencies", "wb"))

print("Getting most common")
words, count = [], []
most_common = art_freq.most_common()

i = 0
while len(words) < 3000:
    w, c = most_common[i]
    if w.isalpha():
        words.append(w)
        count.append(c)
    i += 1

print("Word", words[0], count[0])

print("Plotting words and counts")
plt.bar([i for i in range(len(count))], count)
xticks = [i for i in np.arange(0, 3000, 100)]
tick_words = [words[i] for i in xticks]

plt.title("Article Word Counts")
plt.xticks(xticks, tick_words)
plt.show()
