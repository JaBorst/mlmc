import numpy as np

def load_glove(embedding="/disk1/users/jborst/Data/Embeddings/glove/en/glove.6B.50d_small.txt"):
    glove = np.loadtxt(embedding, dtype='str', comments=None)
    glove = glove[np.unique(glove[:,:1],axis=0, return_index=True)[1]]
    words = glove[:, 0]
    weights = glove[:, 1:].astype('float')
    weights = np.vstack((
                            np.array([0]* len(weights[1])), # the vector for the masking
                            weights,
                            np.mean(weights, axis=0)), # the vector for the masking)
    )
    words = words.tolist()+["<UNK_TOKEN>"]
    vocabulary = dict(zip(words,range(1,len(words)+1)))
    return weights, vocabulary
