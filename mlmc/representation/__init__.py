"""
Functions and classes for loading and using numerical text representations like word embeddings
and language models in an automated fashion.

Most of the time this module will only be used implicitly when creating an instance of an classification model.

## mlmc.representation.get

The main function of `mlmc.representation` is `mlmc.representation.get`. It is a wrapper around downloading,formatting
 and initializting word sequence embedding techniques.
 It works as a string lookup and should support  all model names found on
 https://huggingface.co/models (currently only tested for  common non-community models)
 or one of the keywords ["glove50", "glove100", "glove200" and "glove300"] for static
 embeddings.

 The function always returns an embedding object that inherits from `torch.nn.Module` (so it can be used in
 your own model architectures as layers) and the corresponding tokenizer, so that calling both sequentially will result
in the embedded version of your string, but gives you the freedom to use your own tokenizer ( although not recommended in
case you are using the language models).

Getting an embedding example

```python
import mlmc
embedder, tokenizer = mlmc.representation.get("bert-base-uncased")
embedding_tensor = embedder(tokenizer("Some text you want to embed"))
```
"""

from .representations import load_static, get_transformer, map_vocab,get_embedding,get, is_transformer
from .character import charindex
from .labels import makesequencelabels, schemetransformer, to_scheme, makemultilabels
from .postprocessing_vectors import postprocess_embedding
from .label_embeddings import get_word_embedding_mean, get_lm_generated, get_lm_repeated
from .embedder import Embedder

from .output_transformations import threshold_mcut, threshold_hard