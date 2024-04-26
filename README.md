# GermanToEnglishModel
Code to train a model with PyTorch
Uses sequence-to-sequence learning to translate from German to English.
This makes use of two RNN layers, an "encoder" and a "decoder" to translate a message.
The encoder layer takes the sentence in German and condenses it into a vector. And the
decoder layer will upfold the meaning contained in the vector into English.
