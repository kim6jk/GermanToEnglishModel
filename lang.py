from clean_text import load_clean_sentences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from decoder import device, MAX_WORDS
import numpy as np

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2 #SOS & EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            #Add to word2index dict - key = word, value = current count of words
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            #Opposite of word2index
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def loadLangs(lang1, lang2, filename, reverse=False):
    dataset = load_clean_sentences('pairs/english-german-whole.pkl', reverse)

    if reverse:
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    # Add words to lang objects
    for i in range(len(dataset)):
        input_lang.addSentence(dataset[i, 0])
        output_lang.addSentence(dataset[i, 1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, list(dataset)



def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang,  sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

# def tensorFromPair(pair):
#     input_tensor = tensorFromSentence(input_lang, pair[0])
#     output_tensor = tensorFromSentence(output_lang, pair[1])

def get_dataloader(batch_size):
    input_lang, output_lang, pairs = loadLangs('eng', 'deu', 'pairs/english-german-training.pkl', True)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_WORDS), dtype=np.int32)
    target_ids = np.zeros((n, MAX_WORDS), dtype=np.int32)

    for index, (input, target) in  enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, input)
        tgt_ids = indexesFromSentence(output_lang, target)

        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)

        input_ids[index, :len(inp_ids)] = inp_ids
        target_ids[index, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device), torch.LongTensor(target_ids).to(device))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return input_lang, output_lang, train_dataloader