import re
import string
from unicodedata import normalize
from numpy import array
from pickle import load
from pickle import dump
import numpy as np

def load_doc(filename):
    #Open files containing English - German pairs
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text

def split_pairs(document):
    #Split the Pair document into lines
    lines = document.strip().split('\n')
    #Split the pairs
    pairs = [line.split('\t') for line in lines]
    return pairs

def clean(lines):
    cleaned = []
    #Compile regex to use for filtering
    regex_printable = re.compile('[^%s]' % re.escape(string.printable))
    #Create table to be used for removing puncutation
    table = str.maketrans('', '', string.punctuation)

    for pair in lines:
        cleaned_pair = []
        for line in pair:
            #Normalize the Unicode chars
            line = normalize("NFD", line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')

            line = line.split()
            #Make lower case
            line = [word.lower() for word in line]
            #Remove puncutation
            line = [word.translate(table) for word in line]
            #Use regex to remove non-printable chars
            line = [regex_printable.sub('', word) for word in line]
            #Remove numbers
            line = [word for word in line if word.isalpha()]
            #Store the cleaned pair
            cleaned_pair.append(' '.join(line))
        cleaned.append(cleaned_pair)
    
    return array(cleaned)

def save_cleaned_data(cleaned_pairs, filename):
    dump(cleaned_pairs, open(filename, 'wb'))
    print('Saved: %s' % filename)

def get_max_words(filename):
    dataset = load_clean_sentences('pairs/english-german-whole.pkl', False)
    max_length = 0

    for i in range(len(dataset)):
        eng_sentence_length = len(dataset[i, 0].split(' '))
        deu_sentence_length = len(dataset[i, 1].split(' '))
        if(eng_sentence_length > max_length):
            max_length = eng_sentence_length
        if(deu_sentence_length > max_length):
            max_length = deu_sentence_length

    return max_length

#Load cleaned dataset
def load_clean_sentences(filename, reverse=False):
    dataset = load(open(filename, 'rb'))
    if reverse:
        dataset = np.fliplr(dataset)

    return dataset
