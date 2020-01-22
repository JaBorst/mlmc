import torch
from reformer_pytorch import ReformerLM

import mlmc.helpers.embeddings.representations

model = ReformerLM(
    num_tokens= 20000,
    emb = 128,
    depth = 4,
    max_seq_len = 8192,
    heads = 8,
    lsh_dropout = 0.1,
    causal = True,        # auto-regressive or not
    bucket_size = 64,     # average size of qk per bucket, 64 was recommended in paper
    n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
    ff_chunks = 200,      # number of chunks for feedforward layer, make higher if there are memory issues
    weight_tie = False,   # tie parameters of each layer for no memory per additional depth
    attn_chunks = 8,        # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
    use_full_attn = False   # use full self attention, for comparison
).cuda()


import mlmc
from tqdm import tqdm
data = mlmc.data.get_dataset("blurbgenrecollection", type=mlmc.data.MultiLabelDataset, ensure_valid=False, valid_split=0.25, target_dtype=torch._cast_Float)
weights, vocabulary = mlmc.helpers.load_static(embedding="/disk1/users/jborst/Data/Embeddings/fasttext/static/en/wiki-news-300d-10k.vec")
for x in tqdm(torch.utils.data.DataLoader(data["train"], batch_size=50)):
    num = mlmc.helpers.embeddings.representations.map_vocab(x["text"], vocabulary, maxlen=256).long().cuda()
    y = model(num) # (1, 8192, 20000)
    y.sum().backward()