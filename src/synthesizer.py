from nltk import pos_tag
import nltk
from nltk import word_tokenize
from numpy.random import uniform
import pickle
import src.dataprocessing.preprocessing as preprocess
from collections import defaultdict
import random


class Synthesizer:
    def __init__(self):
        self.data = []  # preprocessing.load_data()

    def tagArticle(self, article):
        """
        Take an article text and creates POS tags
        :param article:
        :return: tagged tokenized words
        """
        if type(article) is str:
            return pos_tag(word_tokenize(str(article)))
        if type(article) is list:
            return pos_tag(article)

    def removePOS(self, tagged_words):
        """
        Removes part-of-speech from the list of tuple of words.
        :param tagged_words:
        :return: words without NN, VB, JJ
        """
        new_words = []
        removed_count = 0
        for word in tagged_words:
            if 'NN' != word[1] and 'JJ' != word[1] and 'VB' != word[1]:
                new_words.append(word)
            else:
                removed_count += 1

        return new_words, removed_count

    def sampleDistribution(self, pos: str, dist_dict: dict):
        """
        Given a certain Part-Of-Speech... NN, JJ or VB. Method will sample from a uniform distribution then use the
        value to sample from a word distribution.
        :param pos: part-of-speech : NN / JJ / VB
        :return: string : word
        """
        s = random.random()

        def sample(dist: list, s: float):
            for i, v in enumerate([a[1] for a in dist]):
                s -= v
                if s < 0:
                    return dist[i][0]

        return sample(dist_dict[pos], s)

    def synthesizeArticle(self, article):
        tagged_article = self.tagArticle(article)
        freq_dist = self.load_freq()
        new_article = ""
        for word, pos in tagged_article:
            if pos in ["NN", "JJ", "VB"]:
                new_article += self.sampleDistribution(pos, freq_dist) + " "
            else:
                new_article += word + " "

        return new_article

    def load_freq(self):
        freq_dist = pickle.load(open("../data/all_articles_frequencies", mode="rb"))
        freq_dist = nltk.FreqDist({k: v for k, v in freq_dist.items() if k.isalpha()})

        preprocess.remove_stopwords(freq_dist)
        bag_of_words = synth.tagArticle([k for k, v in freq_dist.items() if v > 700])

        frequencies = defaultdict(list)

        for word, pos in bag_of_words:
            frequencies[pos] += [[word, freq_dist[word]]]

        for v in frequencies.values():
            total = sum([a[1] for a in v])
            for a in v:
                a[1] = a[1] / total
        return frequencies


article = "WASHINGTON  —   Congressional Republicans have a new fear when it comes to their health care lawsuit " \
          "against the Obama administration: They might win. The incoming Trump administration could choose to " \
          "no longer defend the executive branch against the suit, which challenges the administration’s " \
          "authority to spend billions of dollars on health insurance subsidies for and Americans, handing House " \
          "Republicans a big victory on issues. But a sudden loss of the disputed subsidies could conceivably " \
          "cause the health care program to implode, leaving millions of people without access to health " \
          "insurance before Republicans have prepared a replacement. That could lead to chaos in the insurance " \
          "market and spur a political backlash just as Republicans gain full control of the government. To stave " \
          "off that outcome, Republicans could find themselves in the awkward position of appropriating huge sums " \
          "to temporarily prop up the Obama health care law, angering conservative voters who have been demanding " \
          "an end to the law for years. In another twist, Donald J. Trump’s administration, worried about " \
          "preserving executive branch prerogatives, could choose to fight its Republican allies in the House on " \
          "some central questions in the dispute. Eager to avoid an ugly political pileup, Republicans on Capitol " \
          "Hill and the Trump transition team are gaming out how to handle the lawsuit, which, after the election, " \
          "has been put in limbo until at least late February by the United States Court of Appeals for the District " \
          "of Columbia Circuit. They are not yet ready to divulge their strategy. “Given that this pending " \
          "litigation involves the Obama administration and Congress, it would be inappropriate to comment,” said " \
          "Phillip J. Blando, a spokesman for the Trump transition effort. “Upon taking office, the Trump " \
          "administration will evaluate this case and all related aspects of the Affordable Care Act. ” In a " \
          "potentially decision in 2015, Judge Rosemary M. Collyer ruled that House Republicans had the standing " \
          "to sue the executive branch over a spending dispute and that the Obama administration had been " \
          "distributing the health insurance subsidies, in violation of the Constitution, without approval from " \
          "Congress. The Justice Department, confident that Judge Collyer’s decision would be reversed, quickly " \
          "appealed, and the subsidies have remained in place during the appeal. In successfully seeking a " \
          "temporary halt in the proceedings after Mr. Trump won, House Republicans last month told the court " \
          "that they “and the  ’s transition team currently are discussing potential options for resolution of " \
          "this matter, to take effect after the  ’s inauguration on Jan. 20, 2017. ” The suspension of the case, " \
          "House lawyers said, will “provide the   and his future administration time to consider whether to " \
          "continue prosecuting or to otherwise resolve this appeal. ” Republican leadership officials in the " \
          "House acknowledge the possibility of “cascading effects” if the   payments, which have totaled an " \
          "estimated $13 billion, are suddenly stopped. Insurers that receive the subsidies in exchange for " \
          "paying costs such as deductibles and   for eligible consumers could race to drop coverage since they " \
          "would be losing money. Over all, the loss of the subsidies could destabilize the entire program and " \
          "cause a lack of confidence that leads other insurers to seek a quick exit as well. Anticipating that " \
          "the Trump administration might not be inclined to mount a vigorous fight against the House Republicans " \
          "given the  ’s dim view of the health care law, a team of lawyers this month sought to intervene in " \
          "the case on behalf of two participants in the health care program. In their request, the lawyers " \
          "predicted that a deal between House Republicans and the new administration to dismiss or settle the " \
          "case “will produce devastating consequences for the individuals who receive these reductions, as well " \
          "as for the nation’s health insurance and health care systems generally. ” No matter what happens, " \
          "House Republicans say, they want to prevail on two overarching concepts: the congressional power of " \
          "the purse, and the right of Congress to sue the executive branch if it violates the Constitution " \
          "regarding that spending power. House Republicans contend that Congress never appropriated the money " \
          "for the subsidies, as required by the Constitution. In the suit, which was initially championed by " \
          "John A. Boehner, the House speaker at the time, and later in House committee reports, Republicans " \
          "asserted that the administration, desperate for the funding, had required the Treasury Department " \
          "to provide it despite widespread internal skepticism that the spending was proper. The White House " \
          "said that the spending was a permanent part of the law passed in 2010, and that no annual " \
          "appropriation was required  — even though the administration initially sought one. Just as important " \
          "to House Republicans, Judge Collyer found that Congress had the standing to sue the White House on " \
          "this issue  —   a ruling that many legal experts said was flawed  —   and they want that precedent to be set to restore congressional leverage over the executive branch. But on spending power and standing, the Trump administration may come under pressure from advocates of presidential authority to fight the House no matter their shared views on health care, since those precedents could have broad repercussions. It is a complicated set of dynamics illustrating how a quick legal victory for the House in the Trump era might come with costs that Republicans never anticipated when they took on the Obama White House."

if __name__ == "__main__":
    synth = Synthesizer()
    print(synth.synthesizeArticle(article))
