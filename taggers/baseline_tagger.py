#import bz2
import copy

# Add path to the data
import sys
sys.path.insert(3, '../runner/')

def read_data(source):
    sentence = []
    for line in source:
        line = line.split()
        if(line == []):
            if not(sentence == []):
                yield sentence
                sentence = []
            else:
                continue
        elif (line[0] == '#'):
            continue
        else:
            tag = (line[1], line[3])
            sentence.append(tag)


with open('train.conllu', 'rt', encoding="utf-8") as source:
    train_data = list(read_data(source))

with open('dev.conllu', 'rt', encoding="utf-8") as source:
    dev_data = list(read_data(source))

## Making vocabularies
PAD = '<pad>'
UNK = '<unk>'


def make_vocabs(gold_data):
    vocab_words = {PAD: 0, UNK: 1}
    vocab_tags = {PAD: 0}
    words_index = 2
    tags_index = 1
    for sentence in gold_data:
        for elem in sentence:
            if elem[0] not in vocab_words:
                vocab_words[elem[0]] = words_index
                words_index += 1
            if elem[1] not in vocab_tags:
                vocab_tags[elem[1]] = tags_index
                tags_index += 1
    return vocab_words, vocab_tags


# Tagger interface
class Tagger(object):
    def featurize(self, words, i, pred_tags):
        raise NotImplementedError

    def predict(self, words):
        raise NotImplementedError



def accuracy(tagger, gold_data):
    corr = 0
    total = 0
    for elem in gold_data:
        sen = []
        for word in elem:
            sen.append(word[0])
        pred = tagger.predict(sen)
        for i in range(len(sen)):
            if(pred[i] == elem[i][1]):
                corr += 1
            total += 1
    return corr/total
