from tokenizers import Tokenizer
class enc:
    def __init__(self, tokens, ids):
        self.tokens=tokens
        self.ids=ids

class WhiteSpaceTokenizer(Tokenizer):
    def __init__(self, vocabulary):
        super(self, WhiteSpaceTokenizer).__init__()
        self.vocabulary=vocabulary
    def encode(self, s):
        tokens=s.split()

        return(enc())