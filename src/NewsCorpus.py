import nltk.data
from nltk.tokenize import *

from nltk.corpus.reader.api import *


class NewsCorpusReader(CorpusReader):
    """
    CorpusReader based on the NLTK PlaintextCorpusReader
    """


    def __init__(self, root, fileids,
                 word_tokenizer=WordPunctTokenizer(),
                 sent_tokenizer=nltk.data.LazyLoader(
                     'tokenizers/punkt/english.pickle'),
                 para_block_reader=read_blankline_block,
                 encoding='utf8'):
        """
        Construct a new plaintext corpus reader for a set of documents
        located at the given root directory.  Example usage:

            >>> root = '/usr/local/share/nltk_data/corpora/webtext/'
            >>> reader = PlaintextCorpusReader(root, '.*\.txt') # doctest: +SKIP

        :param root: The root directory for this corpus.
        :param fileids: A list or regexp specifying the fileids in this corpus.
        :param word_tokenizer: Tokenizer for breaking sentences or
            paragraphs into words.
        :param sent_tokenizer: Tokenizer for breaking paragraphs
            into words.
        :param para_block_reader: The block reader used to divide the
            corpus into paragraph blocks.
        """
        CorpusReader.__init__(self, root, fileids, encoding)
        self._word_tokenizer = word_tokenizer
        self._sent_tokenizer = sent_tokenizer
        self._para_block_reader = para_block_reader

    def raw(self, fileids=None):
        """
        :return: the given file(s) as a single string.
        :rtype: str
        """
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, string_types):
            fileids = [fileids]
        raw_texts = []
        for f in fileids:
            _fin = self.open(f)
            raw_texts.append(_fin.read())
            _fin.close()
        return concat(raw_texts)

    def words(self, fileids=None):
        """
        :return: the given file(s) as a list of words
            and punctuation symbols.
        :rtype: list(str)
        """
        return concat([self.CorpusView(path, self._read_word_block, encoding=enc)
                       for (path, enc, fileid)
                       in self.abspaths(fileids, True, True)])

    def sents(self, fileids=None):
        """
        :return: the given file(s) as a list of
            sentences or utterances, each encoded as a list of word
            strings.
        :rtype: list(list(str))
        """
        if self._sent_tokenizer is None:
            raise ValueError('No sentence tokenizer for this corpus')

        return concat([self.CorpusView(path, self._read_sent_block, encoding=enc)
                       for (path, enc, fileid)
                       in self.abspaths(fileids, True, True)])

    def paras(self, fileids=None):
        """
        :return: the given file(s) as a list of
            paragraphs, each encoded as a list of sentences, which are
            in turn encoded as lists of word strings.
        :rtype: list(list(list(str)))
        """
        if self._sent_tokenizer is None:
            raise ValueError('No sentence tokenizer for this corpus')

        return concat([self.CorpusView(path, self._read_para_block, encoding=enc)
                       for (path, enc, fileid)
                       in self.abspaths(fileids, True, True)])

    def articles(self, fileids=None):
        """
        :return: the given file(s) as a list of
            sentences or utterances, each encoded as a list of word
            strings.
        :rtype: list(list(str))
        """
        if self._sent_tokenizer is None:
            raise ValueError('No sentence tokenizer for this corpus')

        return concat([self.CorpusView(path, self._read_sent_block, encoding=enc)
                       for (path, enc, fileid)
                       in self.abspaths(fileids, True, True)])

    def _read_word_block(self, stream):
        words = []
        for i in range(20):  # Read 20 lines at a time.
            words.extend(self._word_tokenizer.tokenize(stream.readline()))
        return words

    def _read_sent_block(self, stream):
        sents = []
        for para in self._para_block_reader(stream):
            sents.extend([self._word_tokenizer.tokenize(sent)
                          for sent in self._sent_tokenizer.tokenize(para)])
        return sents

    def _read_articles_block(self, stream):
        articles = []
        for para in self._para_block_reader(stream):
            articles.extend([article for article in self._sent_tokenizer.tokenize(para)])
        return articles

    def _read_para_block(self, stream):
        paras = []
        for para in self._para_block_reader(stream):
            paras.append([self._sent_tokenizer.tokenize(sent)
                          for sent in self._sent_tokenizer.tokenize(para)])
            # paras.append([sent for sent in self._sent_tokenizer.tokenize(para)])
        return paras
