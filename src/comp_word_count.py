import os
import pickle
import nltk

CURRENT_DIR = os.getcwd()
DATA_DIR = ""
if "src" in CURRENT_DIR:
    DATA_DIR = CURRENT_DIR.replace("src", "data")
elif CURRENT_DIR.split('/')[-1] == "cs175-project":
    DATA_DIR = CURRENT_DIR + "/data"

print("Loading word frequencies from real articles..")
real_art_freq = pickle.load(open(DATA_DIR + "/all_articles_frequencies", "rb"))
real_art_freq = {k.lower(): v for k, v in real_art_freq.items()}

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

real = [[w, real_art_freq[w]] for w, c in most_common[:10]]
fake = [[w, c] for w, c in most_common[:10]]
print("REAL")
for r in real:
    print(r)

print("FAKE")
for f in fake:
    print(f)
