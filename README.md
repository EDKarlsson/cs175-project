# CS175 Project README

### LINKS
- Google Drive	https://drive.google.com/drive/u/1/folders/0AHIZLxFXnw36Uk9PVA
- Github	https://github.com/EDKarlsson/cs175-project
- Kaggle News Data Set 1	https://www.kaggle.com/snapcrack/all-the-news/data
- Automatic deception detection: Methods for finding fake news	http://onlinelibrary.wiley.com/doi/10.1002/pra2.2015.145052010082/full
- Fake News Data Set	https://www.kaggle.com/mrisdal/fake-news
- POS Tags  https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
- LSTM 1 - http://colah.github.io/posts/2015-08-Understanding-LSTMs
- RNN - http://karpathy.github.io/2015/05/21/rnn-effectiveness/
- Gen Text W/ RNN - https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/


## Pipeline
Data:
- For each category:
    - Create POS. word distributions

Synthesizer:
    - Takes a random article
    - Tags words
    - Removes nouns, adjectives, verbs
    - Samples from distribution of words (scipy)