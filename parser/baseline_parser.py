#import bz2
import copy

# Add path to the data
import sys
sys.path.insert(4, '../runner/')

def read_data(source):
    i = 0
    sen = [('<root>', '<root>', 0)]
    for line in source:
        line = line.split()
        if len(line) == 0:
            if(sen != []):
                yield sen
            sen = [('<root>', '<root>', 0)]
        elif line[0] == "#" or line[6] == "_":
            continue
        else:
            sen.append((line[1], line[3], int(line[6])))

with open('train.conllu', 'rt') as source:
    train_data = list(read_data(source))

with open('dev.conllu', 'rt') as source:
    dev_data = list(read_data(source))

##  Make vocabularies for the parsers
PAD = '<pad>'
UNK = '<unk>'

def make_vocabs(gold_data):
    vocab_words = {PAD: 0, UNK: 1}
    vocab_tags = {PAD: 0}
    for sentence in gold_data:
        for word, tag, _ in sentence:
            if word not in vocab_words:
                vocab_words[word] = len(vocab_words)
            if tag not in vocab_tags:
                vocab_tags[tag] = len(vocab_tags)
    return vocab_words, vocab_tags

## Validation
def uas(parser, gold_sentences):
    correct = 0
    total = 0
    for sentence in gold_sentences:
        words, tags, gold_heads = zip(*sentence)
        pred_heads = parser.predict(words, tags)
        for gold, pred in zip(gold_heads[1:], pred_heads[1:]):
            correct += int(gold == pred)
            total += 1
    return correct / total

## Bad standard so can be improved by unpacking instead of copy + tuple "cast"
class Parser():

    MOVES = tuple(range(3))

    SH, LA, RA = MOVES  # Parser moves are specified as integers.

    @staticmethod
    def initial_config(num_words):
        return (0, [], [0]*num_words)

    @staticmethod
    def valid_moves(config):
        valid_moves = []
        if (config[0] < len(config[2])):
            valid_moves.append(0)
        if (len(config[1]) > 2):
            valid_moves.append(1)
        if (len(config[1]) > 1):
            valid_moves.append(2)
        return valid_moves

    @staticmethod
    def next_config(config, move):
        next_config = list(copy.deepcopy(config))
        if (move == 0):
            next_config[1].append(config[0])
            next_config[0] += 1
        elif (move == 1):
            next_config[2][next_config[1][-2]] = next_config[1][-1]
            del next_config[1][-2]
        elif (move == 2):
            next_config[2][next_config[1][-1]] = next_config[1][-2]
            del next_config[1][-1]
        else:
            print("Too moves high number in moves !!! Custom print!")

        return tuple(next_config)

    @staticmethod
    def is_final_config(config):
        if (len(config[1]) == 1 and len(config[2]) == config[0]):
            return True
        else:
            return False

    # non-static part

    def featurize(self, words, tags, config):
        raise NotImplementedError

    def predict(self, words, tags):
        raise NotImplementedError

## Same as above, bad standard
def oracle_moves(gold_heads):
    parser = Parser()
    config = parser.initial_config(len(gold_heads))
    next_config = config[:]
    while(parser.is_final_config(next_config) != True):
        config = next_config
        if(len(config[1]) < 2):
            move = 0
            next_config = parser.next_config(config, move)
            yield config, move
        elif(config[1][-1] == gold_heads[config[1][-2]] and 1 in parser.valid_moves(config)):
            move = 0
            if(config[2].count(config[1][-2]) == gold_heads.count(config[1][-2])):
                move = 1

            next_config = parser.next_config(config, move)
            yield config, move
        elif(config[1][-2] == gold_heads[config[1][-1]] and 2 in parser.valid_moves(config) and (config[0] == len(config[2]) or len(config[1]) > 2 )):# last was config[2]
            move = 0
            if(config[2].count(config[1][-1]) == gold_heads.count(config[1][-1])):
                move = 2

            next_config = parser.next_config(config, move)
            yield config, move
        else:
            move = 0
            next_config = parser.next_config(config, move)
            yield config, move

def samples(gold_data, parser, n_epochs=1):
    for _ in range(n_epochs):
        for sentence in gold_data:
            config = parser.initial_config(len(gold_data))
            word_list = []
            tag_list = []
            gold_heads = []
            for vector in sentence:
                word_list.append(vector[0])
                tag_list.append(vector[1])
                gold_heads.append(vector[2])

            for i, (config, move) in enumerate(oracle_moves(gold_heads)):
                feature = parser.featurize(word_list, tag_list, config)
                yield feature, move
