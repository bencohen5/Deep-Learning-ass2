import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
import math
import torch.nn.functional as F
import sys
import time
import random
import matplotlib.pyplot as plt


class NNE(nn.Module):
    def __init__(self, emb_dim, hidden_size, window_len, train_data, batch_size):
        super(NNE, self).__init__()
        vocab_size = len(train_data.word_idx)
        tags_size = len(train_data.tags_idx)
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.embeddings.shape = torch.Tensor(batch_size, window_len * emb_dim)
        self.linear1 = nn.Linear(window_len * emb_dim, hidden_size)
        self.uniform(self.linear1, window_len * emb_dim, hidden_size)
        self.batch1 = nn.BatchNorm1d(emb_dim)
        self.linear2 = nn.Linear(hidden_size, tags_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs):
        embeds = self.embeddings(torch.tensor(inputs)).view(self.embeddings.shape.size())
        out = self.linear1(self.dropout(embeds)).tanh()
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def epsilon(self, m, n):
        return math.sqrt(6) / math.sqrt(m + n)

    def uniform(self, hidden, n, m):
        tanh_epsilon = self.epsilon(m, n)
        """
        init.uniform_(hidden, -tanh_epsilon, tanh_epsilon)
        """


class NNTE(nn.Module):
    def __init__(self, emb_dim, hidden_size, window_len, train_data, batch_size, prefix_len, suffix_len):
        super(NNTE, self).__init__()
        vocab_size = len(train_data.word_idx)
        tags_size = len(train_data.tags_idx)
        self.embed_Word = nn.Embedding(vocab_size, emb_dim)
        self.embed_Word.shape = torch.Tensor(batch_size, window_len * emb_dim)
        self.embed_pref = nn.Embedding(prefix_len, emb_dim)
        self.embed_pref.shape = torch.Tensor(batch_size, window_len * emb_dim)
        self.embed_suff = nn.Embedding(suffix_len, emb_dim)
        self.embed_suff.shape = torch.Tensor(batch_size, window_len * emb_dim)
        self.linear1 = nn.Linear(window_len * emb_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, tags_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x_words = []
        x_prefix = []
        x_suffix = []
        for words, suf, pre in x:
            x_words.append(words)
            x_prefix.append(pre)
            x_suffix.append(suf)
        h1Words = self.embed_Word(torch.tensor(x_words)).view(self.embed_Word.shape.size())
        h1Prefix = self.embed_pref(torch.tensor(x_prefix)).view(self.embed_pref.shape.size())
        h1Suffix = self.embed_suff(torch.tensor(x_suffix)).view(self.embed_suff.shape.size())
        avg = (h1Words + h1Prefix + h1Suffix) / 3
        h2 = self.linear1(self.dropout(avg)).tanh()
        out = self.linear2(h2)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


class Trainer():
    def __init__(self, model, epoch, lr, train_data, batch_len, predictor):
        self.model = model
        self.epoch = epoch
        self.predictor = predictor
        self.model_input = train_data.model_inputs
        self.word_idx = train_data.word_idx
        self.tag_idx = train_data.tags_idx
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
        self.batch_size = batch_len

    def train(self):
        for epoch in range(self.epoch):
            running_loss = 0.0
            before = time.time()
            x = []
            y = []
            j = 100
            random.shuffle(self.model_input)
            for i, (words, tag) in enumerate(self.model_input):
                x.append(words)
                y.append(tag)
                if i % self.batch_size == self.batch_size - 1:
                    running_loss += self.train_model(x, y)
                    x = []
                    y = []
                    j -= 1
                    if j == 0:
                        j = 100
                        print('[%d %d] loss: %.3f takes %f sec' % (epoch, i, running_loss, time.time() - before))
                        before = time.time()
                        running_loss = 0.0
            self.predictor.predict()

    def train_model(self, words, tag):
        running_loss = 0.0
        running_loss += self.train_net(words, tag)
        return running_loss

    def train_net(self, inputs, label):
        self.optimizer.zero_grad()
        output = self.model(inputs)
        label = torch.tensor(label)
        loss = self.criterion(output, label)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class Predictor():
    def __init__(self, model_data, t_model, batch, is_ner=False):
        self.data = model_data.model_inputs
        self.tag_model = t_model
        # self.word_model = w_model
        self.word_idx = model_data.word_idx
        self.tags_idx = model_data.tags_idx
        self.total = 0.0
        self.correct = 0.0
        self.batch_size = batch
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_values = []
        self.accuracy_values = []
        self.compare = self.general_accuracy
        if is_ner:
            self.compare = self.ner_accuracy

    def reset(self):
        self.total = 0.0
        self.correct = 0.0

    def predict(self):
        self.reset()
        running_loss = 0.0
        x = []
        y = []
        for i, (words, tags) in enumerate(self.data):
            # words and tags for one sentence
            x.append(words)
            y.append(tags)
            if i % self.batch_size == self.batch_size - 1:
                model_output = self.tag_model(x)
                _, predicted = torch.max(self.tag_model(x), 1)
                label = torch.tensor(y)
                running_loss += self.loss_fn(model_output, label)
                self.calculate_accuracy(label, predicted)
                x = []
                y = []
        self.loss_values.append(running_loss)
        self.accuracy_values.append((self.correct / self.total) * 100)
        print("accuracy : %.3f loss: %.3f" % ((self.correct / self.total) * 100, running_loss))

    def test(self, test_data, path):
        with open(path, 'w+') as test_file:
            x = []
            real_words = []
            end_of_sentences = []
            i = 0
            for words, tag in test_data.model_inputs:
                # end of sentence
                if tag is None:
                    end_of_sentences.append(i)
                    continue
                x.append(words)
                # i added the real word to the tag section
                real_words.append(tag)
                if i % self.batch_size == self.batch_size - 1:
                    _, predicted = torch.max(self.tag_model(x), 1)
                    self.write_to_file(test_data, test_file, real_words, predicted, end_of_sentences)
                    x = []
                    real_words = []
                    i = 0
                    end_of_sentences = []
                else:
                    i += 1

    def write_to_file(self, test_data, test_file, words, y_pred, end_of_sentences):
        for i, (word, tag) in enumerate(zip(words, y_pred)):
            if i in end_of_sentences:
                self.write_word(None, None, test_file)
            val = tag.data.cpu().numpy().item(0)
            tag = test_data.idx_tags[val]
            self.write_word(word, tag, test_file)

    def write_word(self, word, tag, test_file):
        if word is None and tag is None:
            test_file.write("\n")
        else:
            test_file.write(word + " " + tag + "\n")

    def create_graphs(self, folder):
        self.generate_plot(folder, self.loss_values, "loss")
        self.generate_plot(folder, self.accuracy_values, "accuracy")

    def generate_plot(self, folder, y, name):
        plt.figure()
        plt.plot(range(len(y)), y, linewidth=2.0)
        plt.xlabel("Epochs")
        plt.ylabel(name)
        plt.title(name + " vs Epochs")
        plt.savefig(folder + '/' + name + ".png")

    def ner_accuracy(self, y, y_pred):
        if y_pred == self.tags_idx['O'] and y == self.tags_idx['O']:
            return
        self.general_accuracy(y, y_pred)

    def general_accuracy(self, y, predict):
        self.total += 1
        if predict == y:
            self.correct += 1

    def calculate_accuracy(self, y, predict):
        for i in range(len(predict)):
            self.compare(y.data[i], predict.data[i])


class DataModel:
    def __init__(self, model_inputs, word_idx, tags_idx, idx_tags, idx_word):
        self.model_inputs = model_inputs
        self.word_idx = word_idx
        self.tags_idx = tags_idx
        self.idx_tags = idx_tags
        self.idx_word = idx_word

    def add_sufiix_prefix(self, data_model_suffix, data_model_prefix):
        suffix = data_model_suffix.model_inputs
        prefix = data_model_prefix.model_inputs
        merge_list = []
        for (word, tag), (suf, tag_s), (pref, tag_ss) in zip(self.model_inputs, suffix, prefix):
            merge_list.append(((word, suf, pref), tag))
        self.model_inputs = merge_list


class Parser:
    def __init__(self):
        self.suffix = ['er', 'ism', 'ment', 'ness', 'ion', 'ate', 'fy', 'ize', 'en', 'al', 'ic',
                       'ive', 'ful', 'less', 'ly', 'en', 'al', 'ance', 'en', 'able',
                       'ible', 'esque', 'ness', 'or', 'ar', 'ment', 'ity', 'like', 'ing', 'ed']
        self.prefix = ['un', 'non', 'in', 'il', 'ir', 'in', 'de', 'dis', 're', 'pre', 'fore', 'post', 'anti',
                       'pro', 'inter', 'intra', 'mid', 'mal', 'mis', 'out', 'nulti', 'poly', 'semi', 'super',
                       'sub', 'over', 'under', 'uni', 'co', 'ex', 'en']
        self.dev_mode = ''
        self.input_file = ''
        self.model_inputs = []
        self.word_idx = {}
        self.tags_idx = {}
        self.idx_word = {}
        self.idx_tags = {}
        self.test_mode = ''
        self.current_sentence_words = []
        self.current_sentence_tags = []
        self.get_input = self.tag_input
        self.suffix_mode = ''
        self.prefix_mode = ''

    def initialization(self, input_file_name, is_dev, is_test, is_pre_trained, is_suffix, is_preffix):
        self.start_tag = '<s>'
        self.end_tag = '</s>'
        self.dev_mode = is_dev
        self.test_mode = is_test
        self.pre_trained_mode = is_pre_trained
        self.input_file = open(input_file_name)
        self.model_inputs = list()
        if not self.dev_mode and not self.test_mode and not self.pre_trained_mode:
            self.word_idx = {self.start_tag: 0, self.end_tag: 1}
            self.idx_word = {0: self.start_tag, 1: self.end_tag}
            if not self.pre_trained_mode:
                self.add_pattern()
            self.idx_tags = {}
            self.tags_idx = {}
        self.current_sentence_words = [self.start_tag, self.start_tag]
        self.real_words = [self.start_tag, self.start_tag]
        self.current_sentence_tags = [self.start_tag, self.start_tag]
        self.get_input = self.tag_input
        self.suffix_mode = is_suffix
        self.prefix_mode = is_preffix

    def parse_input(self, input_file_name, is_dev=False, is_test=False, is_pre_trained=False, is_suffix=False,
                    is_preffix=False):
        self.initialization(input_file_name, is_dev, is_test, is_pre_trained, is_suffix, is_preffix)
        lines = self.input_file.readlines()
        before = time.time()
        for line in lines:
            line = line.replace("\n", "")
            # end of sentence
            if not line:
                self.add_as_input()
                continue
            if not is_test:
                word, tag = self.split_line(line)
                self.add_to_dict(word, tag)
            else:
                word = line
                # put the real word in tag section because we need it when we
                # write to file (instead of the pattern that appear in word section
                self.add_to_dict(word, word)
        print("loading data took :" + str(time.time() - before) + " seconds")
        self.input_file.close()
        return self.get_data()

    def check_suffix_and_prefix(self, word):
        if (not self.prefix_mode and not self.suffix_mode) or \
                self.is_number(word) or len(word) < 3:
            return word
        if self.suffix_mode:
            return word[-3:]
        return word[:3]

    def add_to_dict(self, word, tag):
        if not self.pre_trained_mode:
            if self.is_number(word):
                word = '^num'
            word = self.check_suffix_and_prefix(word)
            self.current_sentence_words.append(word)
        else:
            word = self.check_suffix_and_prefix(word)
            self.current_sentence_words.append(word.lower())
        self.current_sentence_tags.append(tag)
        if not self.dev_mode and not self.test_mode:
            if word not in self.word_idx and not self.pre_trained_mode:
                self.add_word(word)
            if tag not in self.tags_idx:
                self.idx_tags[len(self.tags_idx)] = tag
                self.tags_idx[tag] = len(self.tags_idx)

    def add_pattern(self):
        patterns = self.suffix + self.prefix + ['Unk', 'num']
        for p in patterns:
            self.add_word('^' + p)

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False

    def catch_pattern(self, w):
        if self.pre_trained_mode:
            return 'UUUNKKK'
        pattern = "^Unk"
        if self.is_number(w):
            return "^num"
        for suf in self.suffix:
            if w.endswith(suf):
                return '^' + suf
        for pre in self.prefix:
            if w.startswith(pre):
                return '^' + pre
        return pattern

    def get_data(self):
        data_model = DataModel(self.model_inputs, self.word_idx, self.tags_idx, self.idx_tags, self.idx_word)
        return data_model

    def add_as_input(self):
        self.current_sentence_words.append(self.end_tag)
        self.current_sentence_words.append(self.end_tag)
        self.current_sentence_tags.append(self.end_tag)
        self.current_sentence_tags.append(self.end_tag)
        self.get_input()
        self.current_sentence_words = [self.start_tag, self.start_tag]
        self.current_sentence_tags = [self.start_tag, self.start_tag]
        self.real_words = [self.start_tag, self.start_tag]

    def tag_input(self):
        if self.dev_mode or self.test_mode or self.pre_trained_mode:
            self.words_idx_checking(self.current_sentence_words)
        for i, word in enumerate(self.current_sentence_words[2:-2]):
            inputs = [self.word_idx[self.current_sentence_words[i]],
                      self.word_idx[self.current_sentence_words[i + 1]],
                      self.word_idx[self.current_sentence_words[i + 2]],
                      self.word_idx[self.current_sentence_words[i + 3]],
                      self.word_idx[self.current_sentence_words[i + 4]]]
            if not self.test_mode:
                output = self.tags_idx[self.current_sentence_tags[i + 2]]
            else:
                output = self.current_sentence_tags[i + 2]
            self.model_inputs.append((inputs, output))
        if self.test_mode:
            self.model_inputs.append((None, None))

    def words_idx_checking(self, words):
        for i, word in enumerate(words):
            if word not in self.word_idx:
                words[i] = self.catch_pattern(word)

    def add_word(self, word):
        self.idx_word[len(self.word_idx)] = word
        self.word_idx[word] = len(self.word_idx)

    def split_line(self, line):
        word_tag = line.split()
        return word_tag[0], word_tag[1]


def train_embedding(is_pos=True):
    pharams = sys.argv
    epoch = 6
    if is_pos:
        epoch = 3
    lr = 1e-3
    batch_size = 159
    parser = Parser()
    train_data_tags_model = parser.parse_input(pharams[2])
    dev_data_tags_model = parser.parse_input(pharams[3], True)
    test_data_tags_model = parser.parse_input(pharams[4], False, True)
    tags_model = NNE(50, 100, 5, train_data_tags_model, batch_size)
    predictor = ""
    if not is_pos:
        predictor = Predictor(dev_data_tags_model, tags_model, batch_size, True)
    else:
        predictor = Predictor(dev_data_tags_model, tags_model, batch_size)
    trainer_tag = Trainer(tags_model, epoch, lr, train_data_tags_model, batch_size, predictor)
    trainer_tag.train()
    predictor.create_graphs(pharams[5])
    predictor.test(test_data_tags_model, pharams[6])


def loadtxt(vocab_file):
    with open(vocab_file) as file:
        vocab_list = []
        for line in file:
            line = line.replace("\n", "")
            if line != "":
                vocab_list.append(line)
        return vocab_list


def pre_trained_embedding(is_pos=True):
    epoch = 4
    lr = 1e-3
    batch_size = 159
    pharams = sys.argv
    word_vectors = pharams[8]
    vocab_input = pharams[9]
    vecs = np.loadtxt(word_vectors)
    vocab = loadtxt(vocab_input)
    word_idx = {word: i for i, word in enumerate(vocab)}
    idx_word = {i: word for i, word in enumerate(vocab)}
    parser = Parser()
    parser.word_idx = word_idx
    parser.idx_word = idx_word
    train_data_tags_model = parser.parse_input(pharams[2], False, False, True)
    dev_data_tags_model = parser.parse_input(pharams[3], True, False, True)
    test_data_tags_model = parser.parse_input(pharams[4], False, True, True)
    tags_model = NNE(50, 100, 5, train_data_tags_model, batch_size)
    tags_model.embeddings.weight.data.copy_(torch.from_numpy(vecs))
    predictor = ""
    if not is_pos:
        predictor = Predictor(dev_data_tags_model, tags_model, batch_size, True)
    else:
        predictor = Predictor(dev_data_tags_model, tags_model, batch_size)
    predictor.predict()
    trainer_tag = Trainer(tags_model, epoch, lr, train_data_tags_model, batch_size, predictor)
    trainer_tag.train()
    predictor.create_graphs(pharams[5])
    predictor.test(test_data_tags_model, pharams[6])


def train_embedding_with_extra(is_pos=True):
    pharams = sys.argv
    epoch = 6
    if is_pos:
        epoch = 2
    lr = 1e-3
    batch_size = 169
    parser_words = Parser()
    parser_prefix = Parser()
    parser_suffix = Parser()
    train_data_tags_model = parser_words.parse_input(pharams[2])
    train_data_tags_model_suffix = parser_suffix.parse_input(pharams[2], False, False, False, True, False)
    train_data_tags_model_prefix = parser_prefix.parse_input(pharams[2], False, False, False, False, True)
    train_data_tags_model.add_sufiix_prefix(train_data_tags_model_suffix, train_data_tags_model_prefix)
    pref_len = len(train_data_tags_model_prefix.word_idx)
    suff_len = len(train_data_tags_model_suffix.word_idx)
    dev_data_tags_model = parser_words.parse_input(pharams[3], True)
    dev_data_tags_model_suffix = parser_suffix.parse_input(pharams[3], True, False, False, True, False)
    dev_data_tags_model_prefix = parser_prefix.parse_input(pharams[3], True, False, False, False, True)
    dev_data_tags_model.add_sufiix_prefix(dev_data_tags_model_suffix, dev_data_tags_model_prefix)
    test_data_tags_model = parser_words.parse_input(pharams[4], False, True)
    test_data_tags_model_suffix = parser_suffix.parse_input(pharams[4], False, True, False, True, False)
    test_data_tags_model_prefix = parser_prefix.parse_input(pharams[4], False, True, False, False, True)
    test_data_tags_model.add_sufiix_prefix(test_data_tags_model_suffix, test_data_tags_model_prefix)
    tags_model = NNTE(50, 100, 5, train_data_tags_model, batch_size, pref_len, suff_len)
    predictor = ""
    if not is_pos:
        predictor = Predictor(dev_data_tags_model, tags_model, batch_size, True)
    else:
        predictor = Predictor(dev_data_tags_model, tags_model, batch_size)
    trainer_tag = Trainer(tags_model, epoch, lr, train_data_tags_model, batch_size, predictor)
    trainer_tag.train()
    predictor.create_graphs(pharams[5])
    predictor.test(test_data_tags_model, pharams[6])


def train_embedding_with_extra_and_pre_trained(is_pos=True):
    pharams = sys.argv
    epoch = 6
    if is_pos:
        epoch = 2
    lr = 1e-3
    batch_size = 100
    word_vectors = pharams[8]
    vocab_input = pharams[9]
    vecs = np.loadtxt(word_vectors)
    vocab = loadtxt(vocab_input)
    word_idx = {word: i for i, word in enumerate(vocab)}
    idx_word = {i: word for i, word in enumerate(vocab)}
    parser_words = Parser()
    parser_prefix = Parser()
    parser_suffix = Parser()
    parser_words.word_idx = word_idx
    parser_words.idx_word = idx_word
    train_data_tags_model = parser_words.parse_input(pharams[2], False, False, True)
    train_data_tags_model_suffix = parser_suffix.parse_input(pharams[2], False, False, False, True, False)
    train_data_tags_model_prefix = parser_prefix.parse_input(pharams[2], False, False, False, False, True)
    train_data_tags_model.add_sufiix_prefix(train_data_tags_model_suffix, train_data_tags_model_prefix)
    pref_len = len(train_data_tags_model_prefix.word_idx)
    suff_len = len(train_data_tags_model_suffix.word_idx)
    dev_data_tags_model = parser_words.parse_input(pharams[3], True, False, True)
    dev_data_tags_model_suffix = parser_suffix.parse_input(pharams[3], True, False, False, True, False)
    dev_data_tags_model_prefix = parser_prefix.parse_input(pharams[3], True, False, False, False, True)
    dev_data_tags_model.add_sufiix_prefix(dev_data_tags_model_suffix, dev_data_tags_model_prefix)
    test_data_tags_model = parser_words.parse_input(pharams[4], False, True, True)
    test_data_tags_model_suffix = parser_suffix.parse_input(pharams[4], False, True, False, True, False)
    test_data_tags_model_prefix = parser_prefix.parse_input(pharams[4], False, True, False, False, True)
    test_data_tags_model.add_sufiix_prefix(test_data_tags_model_suffix, test_data_tags_model_prefix)
    tags_model = NNTE(50, 100, 5, train_data_tags_model, batch_size, pref_len, suff_len)
    tags_model.embed_Word.weight.data.copy_(torch.from_numpy(vecs))
    predictor = ""
    if not is_pos:
        predictor = Predictor(dev_data_tags_model, tags_model, batch_size, True)
    else:
        predictor = Predictor(dev_data_tags_model, tags_model, batch_size)
    trainer_tag = Trainer(tags_model, epoch, lr, train_data_tags_model, batch_size, predictor)
    trainer_tag.train()
    predictor.create_graphs(pharams[5])
    predictor.test(test_data_tags_model, pharams[6])


def main():
    """
    pharam[1]= type of mission : 1 - task1 3 - task 3 41 - task 4 without pre trained 42 - task 4 with pre embedding
    pharam [2] = train.file
    pharam [3] = dev.file
    pharam [4] = test.file
    pharam [5] = graphs output directory
    pharam [6] = prediction output file
    pharam [7] = 1 - pos 2 - ner
    pharam [8] = optional --- wordVectors.txt
    pharam [9] = optional --- vocab.txt
    :return:
    """
    pharams = sys.argv
    if pharams[1] == '1':
        if pharams[7] == '2':
            train_embedding(False)
        else:
            train_embedding()
    elif pharams[1] == '3':
        if pharams[7] == '2':
            pre_trained_embedding(False)
        else:
            pre_trained_embedding()
    elif pharams[1] == '41':
        # with suffix and prefix information
        if pharams[7] == '2':
            train_embedding_with_extra(False)
        else:
            train_embedding_with_extra()
    elif pharams[1] == '42':
        if pharams[7] == '2':
            train_embedding_with_extra_and_pre_trained(False)
        else:
            train_embedding_with_extra_and_pre_trained()


if __name__ == "__main__":
    main()
