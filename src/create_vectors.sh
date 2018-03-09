#!/bin/bash

cd  /Users/dank/git/uci/cs/cs175/cs175-project/src
python -c "from preprocessing import *; create_corpus()"

cp corpus.txt /Users/dank/git/nlp/GloVe/
cd /Users/dank/git/nlp/GloVe/
./create_vectors.sh

cp vectors.txt /Users/dank/git/uci/cs/cs175/cs175-project/data/corpus_vectors.txt

