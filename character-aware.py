"""
To recap, during training we take in each input word by word and look up their word-level representation,
feed them through an RNN to generate a prediction. We then, train our network against the true probability distribution of the label in our vocabulary.

Although these word-level embeddings work great in practice; when we encounter unknown words or rare words our predictions might be not as accurate.

"""



class Char(object):
 
    """ Have a mapping for character to indices and indices to characters"""

    def __init__(self):
 
      
        self.char_list = list("""ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]""")

        self.char2id = dict()  # Converts characters to integers
        self.char2id['<pad>'] = 0  # <pad> token
        self.char2id['{'] = 1  # start of word token
        self.char2id['}'] = 2  # end of word token
        self.char2id['<unk>'] = 3  # <unk> token
        
        # go through all characters and 
        for c in self.char_list:
            self.char2id[c] = len(self.char2id)
        
        
        self.char_pad = self.char2id['<pad>']
        self.char_unk = self.char2id['<unk>']
        self.start_of_word = self.char2id["{"]
        self.end_of_word = self.char2id["}"]
        
        assert self.start_of_word + 1 == self.end_of_word


        self.id2char = {v: k for k, v in self.char2id.items()}  # Converts integers to characters
      
      
    def words_2_indices(self, sents):
        """ Convert list of sentences of words into list of list of list of character indices.
        @param sents (list[list[str]]): sentence(s) in words
        @return word_ids (list[list[list[int]]]): sentence(s) in indices
        """
        return [[[self.char2id.get(c, self.char_unk) for c in ("{" + w + "}")] for w in s] for s in sents]
