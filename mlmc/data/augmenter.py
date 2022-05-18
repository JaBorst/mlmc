import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf

class Augmenter:
    def __init__(self, mode,
                 char_random=False,
                 char_ocr=False,
                 word_random=False,
                 word_contextual=False,
                 wordnet=False,
                 sentence_contextual=False,
                 tfidf=False,
                 word_contextual_representation = 'distilroberta-base',
                 sentence_contextual_representation = 'xlnet-base-cased', ):
        self.mode = mode
        assert mode in ("sequential", "sometimes"), f"Augmentation mode must be in ('sequential', 'sometimes') not {mode}"

        self._config = {
            "augmenter": {
                "sentence_contextual":sentence_contextual,
                "wordnet": wordnet,
                "word_contextual": word_contextual,
                "word_random": word_random,
                "char_random": char_random,
                "char_ocr": char_ocr,
                "tfidf": tfidf,
            },
            "word_contextual_representation": word_contextual_representation,
            "sentence_contextual_representation":sentence_contextual_representation
        }
        if mode == "sequential":
            self.flow = naf.Sequential(self._get_list())
        elif mode == "sometimes":
            self.flow = naf.Sometimes(self._get_list())

    def _get(self, name):
        if name == "char_random":
            return nac.RandomCharAug(aug_word_p = self._config["augmenter"]["char_random"])
        if name == "char_ocr":
            return nac.OcrAug(aug_word_p = self._config["augmenter"]["char_ocr"])
        if name== "word_random":
            return naw.RandomWordAug(aug_p = self._config["augmenter"]["word_random"])
        if name == "word_contextual":
            return naw.ContextualWordEmbsAug(model_path=self._config["word_contextual_representation"],
                                             aug_p = self._config["augmenter"]["word_contextual"])
        if name == "sentence_contextual":
            return nas.ContextualWordEmbsForSentenceAug(model_path=self._config["sentence_contextual_representation"])
        if name == "wordnet":
            return naw.SynonymAug(aug_src='wordnet', aug_p= self._config["augmenter"]["wordnet"])

    def _get_list(self):
        return [self._get(k) for k, v in self._config["augmenter"].items() if v]

    def forward(self, x):
        return self.flow.augment(x)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def generate(self, x, n=10):
        x = [x] if isinstance(x, str) else x
        x = sum([self.flow.augment(s, n=n) for s in x], [])
        return x

    def _linearize_dict(self, d, prefix=""):
        result = []
        for k, v in d.items():
            if isinstance(v, dict):
                result.extend(self._linearize_dict(v, prefix=k))
            else:
                result.append((prefix+k,v))
        return result

    def log_mlflow(self):
        import mlflow
        mlflow.log_params(dict(self._linearize_dict(self._config)))
