from numpy.random import rand
from numpy.random import shuffle
from clean_text import save_cleaned_data
from clean_text import load_clean_sentences


def split_dataset(filename, n_size=0):
    dataset = load_clean_sentences(filename)

    if n_size != 0:
        dataset = dataset[:n_size, :]
        #Shuffle data
        shuffle(dataset)

    training_size = int(len(dataset) * 0.9)

    training, testing = dataset[:training_size], dataset[training_size:]


    save_cleaned_data(dataset, 'pairs/english-german-whole.pkl')
    save_cleaned_data(training, 'pairs/english-german-training.pkl')
    save_cleaned_data(testing, 'pairs/english-german-testing.pkl')
    

