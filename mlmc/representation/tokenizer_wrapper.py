import torch


class TokenizerWrapper():
    """
    A Wrapper around tokenizer functions to provide a unified interface wether using word embeddings
    or transformer models. For transformer mordels it is also possible to ereturn a string of indices
    mapping back to the original words.
    """

    def __init__(self, tokenizer_class, path, cased=False):
        """
        Initializes a tokenizer instance.

        :param tokenizer_class: A model specific tokenizer (see transformers library)
        :param path: A model identifier (see https://huggingface.co/models) or path to config.json file
        :param cased: If True, the case of all words will be kept, else the case will be converted to lowercase
        """
        self.tokenizer = tokenizer_class.from_pretrained(path)

        tmp = self.tokenizer.encode("a")
        if len(tmp) == 3:
            self._bos = [tmp[0]]
            self._eos = [tmp[-1]]
        else:
            self._bos = []
            self._eos = []

        self._umlautmap = {ord(x): y for x, y in zip("öäüÖÄÜ", "oauOAU")}
        self.cased = cased

    def _tokenize_without_index(self, x):
        """
        Wrapper function for shallow tokenization.

        :param x: A string or a list of strings
        :return: A list of tensors containing the token IDs for each list
        """
        x = [x] if isinstance(x, str) else x
        i = self._shallow_tokenize(x)
        return i

    # @timeit
    def _shallow_tokenize(self, x):
        """
        Splits sentences by whitespaces and tokenizes on word level.

        :param x: A list of strings
        :return: A list of tensors containing the token IDs for each list
        """
        return [torch.tensor(x) for x in self.tokenizer.batch_encode_plus(x, pad_to_max_length=False)["input_ids"]]

    def _deep_tokenize(self, x):
        """
        Splits sentences into characters and tokenizes on characters level.

        :param x: A string or a list of strings
        :return: A list of lists of lists each containing the ID of a token.
        """
        return (
        [[self.tokenizer.encode(w, add_special_tokens=False, pad_to_max_length=False) for w in sent] for sent in x])

    def _remove_special_markup(self, text: str):
        """
        Removes special tokens used by language models.

        :param text: A string
        :return: Cleaned string
        """
        # remove special markup
        import re
        text = re.sub('^Ġ', '', text)  # RoBERTa models
        text = re.sub('^##', '', text)  # BERT models
        text = re.sub('^▁', '', text)  # XLNet models
        text = re.sub('</w>$', '', text)  # XLM models
        return text

    def _compare(self, tokens, wp):
        """
        TODO: Documentation

        :param tokens:
        :param wp:
        :return:
        """
        assert len(tokens) > 0, "Empty string input to Embedder"
        tokens_iter = iter(tokens)
        ids = []
        token = next(tokens_iter)
        try:
            for i, w in enumerate(wp):
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
        """
        This is somewhat a critical function. Still does not work perfect. (cmp github.com/flair)
        ToDo: Extensive testing
        Args:
            x:

        Returns:

        """
        x = [x] if isinstance(x, str) else x
        if not self.cased:
            x = [sentence.translate(self._umlautmap).lower() for sentence in x]
        else:
            x = [sentence.translate(self._umlautmap) for sentence in x]

        split_sentences = [w.upper().split() for w in x]  # <- This is not perfect

        tokenized = [self.tokenizer.tokenize(sent) for sent in x]
        ids = [self._compare(s, t) for s, t in zip(split_sentences, tokenized)]
        if not (self._bos == [] or not self.add_special_tokens): ids = [x + 1 for x in ids]

        if self.add_special_tokens:
            tokenized = [torch.tensor(self._bos + self.tokenizer.convert_tokens_to_ids(sent) + self._eos) for sent in
                         tokenized]
        else:
            tokenized = [torch.tensor(self.tokenizer.convert_tokens_to_ids(sent)) for sent in tokenized]

        return tokenized, ids

    def _pad_to_maxlen(self, x, maxlen):
        """
        Pads a 2D-tensor along the second dimension.

        :param x: A two-dimensional tensor
        :param maxlen: The length the tensor will be padded (or cut respectively) to
        :return: A padded tensor of shape (x.shape[0], maxlen)
        """
        r = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded = torch.zeros(r.shape[0], maxlen).long() + self.tokenizer.pad_token_id
        padded[:, :min(maxlen, r.shape[-1])] = r[:, :min(maxlen, r.shape[-1])]
        return padded

    def _ids_to_mask(self, ids, shape):
        """
        Creates a mask for each token.

        :param ids: A tensor containing token IDs
        :param shape: Shape of form (len(ids), len(ids))
        :return: A boolean tensor of shape (len(ids), len(ids))
        """
        mask = torch.zeros(shape)
        for i, sent in enumerate(ids):
            mask[i].scatter_(0, sent, 1.)
        return mask == 1.

    def tokenize(self, x, maxlen=500, return_start=False, pad=True, as_mask=True, add_special_tokens=False):
        """
        Main functionality of tokenizeing according to the tokenizing fcuntion of the current model.

        Args:
            x:  list of strings
            maxlen:  maximum length of token sequence ( longer sequences will be cut)
            return_start: If True returns the start of the words in the input sequence for the transformer embeddings)
            pad: if True the input sequences will be padded to maxlen
            as_mask: if true the start indices mask will be converted to mask
            add_special_tokens: is special_tokens should be added (compare transformers library)

        Returns:

        """
        assert not (
                    return_start and not pad and as_mask), "Returning start ids as mask only possible for padded tokenized output"
        self.add_special_tokens = add_special_tokens
        x = list(x)
        assert isinstance(x[0], str), "Input to tokenizer must be a list of strings!"
        if return_start:
            result, ids = self._tokenize_with_index(x)
            if pad:
                result = self._pad_to_maxlen(result, maxlen)
            if as_mask:
                ids = self._ids_to_mask(ids, result.shape)
            r = (result, ids)
        else:
            # This is the main functionality for models
            r = self._tokenize_without_index(x)
            if pad:
                r = self._pad_to_maxlen(r, maxlen)
        return r

    def __call__(self, *args, **kwargs):
        """
        Method to allow the instance to be called.

        :param args: arguments (see tokenize())
        :param kwargs: keyword arguments (see tokenize())
        :return: Output of tokenize()
        """
        return self.tokenize(*args, **kwargs)
