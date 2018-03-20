# CS175 Project README

### Directory Structure
```
├── NewsCorpus.py
├── comp_word_count.py
├── create_vectors.sh
├── doc_cluster.py
├── fake_real_graph.py
├── models
│   ├── categories.py
│   ├── classifier.py
│   └── kmodel.py
├── preprocessing.py
├── synthesizer.py
├── word_cluster.py
└── word_freq.py
```

## preprocessing.py
Performs text processing for the other models.

## synthesizer.py
Simple text generation synthesizer -- embeds words into articles.

## kmodel
LSTM Model module. Creates or loads models, trains the neural network.

## classifier.py
Uses LSTM and preprocessing, trains a logistic regression classifier and outputs a score.

## doc_cluster.py
Agglomerative clustering of documents for the news articles and graphs it.

## word_cluster.py
Agglomerative clustering of all the words in the news articles and graphs it.

## word_freq.py
Computes the word frequency of real and fake articles then generates a graph.

## comp_word _count.py
Loads both real and fake news articles and prints the count of most common words from fake articles to real articles.

## NewsCorpus.py
Custom PlainTextReader from NLTK that is our news corpus class

## create_vectors.sh
A script that will generate a corpus text then give it to GloVe and generate the co-occurrence vectors.

