from clean_text import *
from split_text import split_dataset
from lang import loadLangs, get_dataloader
from encoder import EncoderRNN
from decoder import device, DecoderRNN
from training import train

# file_name = 'deu.txt'
# document = load_doc(file_name)
# pairs = split_pairs(document)
# clean_pairs = clean(pairs)

# save_cleaned_data(clean_pairs, 'pairs/english-german.pkl')

# split_dataset('pairs/english-german.pkl')

#input_lang, output_lang = loadLangs('eng', 'deu', 'pairs/english-german-whole.pkl', True)

hidden_size = 128
batch_size = 32

input_lang, output_lang, train_dataloader = get_dataloader(batch_size)

encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = DecoderRNN(hidden_size, output_lang.n_words).to(device)

print(device)

train(train_dataloader, encoder, decoder, 80, print_every=1)