# GermanToEnglishModel
Code to train a model with PyTorch
Uses sequence-to-sequence learning to translate from German to English.
This makes use of two RNN layers, an "encoder" and a "decoder" to translate a message.
The encoder layer takes the sentence in German and condenses it into a vector. And the
decoder layer will upfold the meaning contained in the vector into English.

Files:
models - Where the models are saved when checkpointed
	
pairs - Where the English - German sentence pairs are saved
	
clean_text.py - Used to clean/normalize text before training

decoder.py - The decoder RNN

deu.txt - File containing English - German sentences from (http://www.manythings.org/anki/) 
	
encoder.py - The encoder RNN

lang.py - Class used to translate sentences to vectors with one hot encoding
	
main.py - Used to run data setup and training

split_text.py - Used to split the dataset between training data and testing data
	
timer.py - Helper functions to display the duration of the training

training.py - Model training 
