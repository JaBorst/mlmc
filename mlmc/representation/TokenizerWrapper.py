import torch

class TokenizerWrapper():
    def __init__(self, tokenizer_class, path, cased=False):
        self.tokenizer = tokenizer_class.from_pretrained(path)

        tmp = self.tokenizer.encode("a")
        if len(tmp) == 3:
            self._bos = [tmp[0]]
            self._eos = [tmp[-1]]
        else:
            self._bos = []
            self._eos = []

        self._umlautmap = {ord(x):y for x,y in zip("öäüÖÄÜ", "oauOAU")}
        self.cased = cased

    def _tokenize_without_index(self, x):
        x = [x] if isinstance(x, str) else x
        i = self._shallow_tokenize(x)
        return i

    # @timeit
    def _shallow_tokenize(self,x):
        return [torch.tensor(x) for x in self.tokenizer.batch_encode_plus(x, pad_to_max_length=False)["input_ids"]]

    def _deep_tokenize(self,x):
        return ([[self.tokenizer.encode(w, add_special_tokens=False, pad_to_max_length=False) for w in sent] for sent in x])

    def _remove_special_markup(self, text: str):
        # remove special markup
        import re
        text = re.sub('^Ġ', '', text)  # RoBERTa models
        text = re.sub('^##', '', text)  # BERT models
        text = re.sub('^▁', '', text)  # XLNet models
        text = re.sub('</w>$', '', text)  # XLM models
        return text
    def _compare(self, tokens, wp):
        assert len(tokens) > 0, "Empty string input to Embedder"
        tokens_iter = iter(tokens)
        ids = []
        token = next(tokens_iter)
        try:
            for i,w in enumerate(wp):
                if token.startswith(self._remove_special_markup(w).upper()):
                    ids.append(i)
                    token = next(tokens_iter)
        except StopIteration:
            pass
        if token != tokens[-1]:
            print("Alignment mismatch")
        if len(tokens) != len(ids):
            print("Alignment Mismatch between length of ids and length of tokens")
        return torch.tensor(ids)

    def _tokenize_with_index(self, x):
        x = [x] if isinstance(x, str) else x
        if not self.cased:
            x = [sentence.translate(self._umlautmap).lower() for sentence in x]
        else:
            x = [sentence.translate(self._umlautmap) for sentence in x]


        split_sentences = [w.upper().split() for w in x]# <- This is not perfect

        tokenized = [self.tokenizer.tokenize(sent) for sent in x]
        ids = [self._compare(s, t) for s, t in zip(split_sentences,tokenized)]
        if not(self._bos == [] or not self.add_special_tokens): ids = [x+1 for x in ids]

        if self.add_special_tokens:
            tokenized = [torch.tensor(self._bos +self.tokenizer.convert_tokens_to_ids(sent) + self._eos) for sent in tokenized]
        else:
            tokenized = [torch.tensor(self.tokenizer.convert_tokens_to_ids(sent)) for sent in tokenized]

        return tokenized, ids

    def _pad_to_maxlen(self, x, maxlen):
        r = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded = torch.zeros(r.shape[0], maxlen).long() + self.tokenizer.pad_token_id
        padded[:, :min(maxlen, r.shape[-1])] = r[:, :min(maxlen, r.shape[-1])]
        return padded

    def _ids_to_mask(self, ids, shape):
        mask = torch.zeros(shape)
        for i, sent in enumerate(ids):
            mask[i].scatter_(0, sent, 1.)
        return mask == 1.

    def tokenize(self, x, maxlen=500, return_start=False, pad=True, as_mask=True, add_special_tokens=False):
        assert not(return_start and not pad and as_mask), "Returning start ids as mask only possible for padded tokenized output"
        self.add_special_tokens = add_special_tokens
        if return_start:
            result, ids = self._tokenize_with_index(x)
            if pad:
                result = self._pad_to_maxlen(result, maxlen)
            if as_mask:
                ids = self._ids_to_mask(ids, result.shape)
            r = (result,ids)
        else:
            r = self._tokenize_without_index(x)
            if pad:
                r = self._pad_to_maxlen(r, maxlen)
        return r

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)



