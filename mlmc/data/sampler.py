from numpy.random import choice


def sampler(dataset, fraction=None, absolute=None):
    '''Sample a Random subsample of fixed size or fixed fraction of a dataset.'''
    n_samples = absolute if absolute is not None else  int(fraction*len(dataset))
    ind = choice(range(len(dataset)), n_samples)
    x = [dataset.x[i] for i in ind]
    y = [dataset.y[i] for i in ind]
    return type(dataset)(x=x, y=y, classes=dataset.classes, target_dtype=dataset.target_dtype)


def successive_sampler(dataset, classes, separate_dataset):
    """Return an iterable of datasets sampled from dataset... """
   
    n_result = []
    n_idx = []
    n_ind = []
    already_select_id = set()
    already_select_class = {}
    n_samples =  np.round(len(dataset)/separate_dataset).astype(np.int64)

    for x in range(0, separate_dataset):
        # extend classes to use
        # sample from existing classes/delete from whole dataset to sample
        c_ind = choice(range(len(classes)), np.round(len(classes)/2).astype(np.int64))

        selectedKeys = list() 
        
        # Iterate over the dict and put to be deleted keys in the list
        for index, (key, value) in enumerate(classes.items()):
            if(index in c_ind):
                print(index, key, value)
                already_select_class[key] = value
                selectedKeys.append(key)
 
        #Iterate over the list and delete corresponding key from dictionary
        for key in selectedKeys:
            if key in classes :
                del classes[key]
        
        l_list = dataset.y
        
        #Object to store which data point belong to already defined classes
        candidate_idx = []

        for index, (c_list) in enumerate(dataset.y):
            intersect = set(c_list).intersection(set(already_select_class.keys()))
            if len(intersect) > 0:
                candidate_idx.append(index)
                l_list[index] = list(intersect)


        ind = choice(list(set(candidate_idx) - already_select_id), n_samples)

        already_select_id = set(ind).union(already_select_id)



        x = [dataset.x[i] for i in list(already_select_id)]
        y = [l_list[i] for i in list(already_select_id)]

        x_new = [dataset.x[i] for i in ind]
        y_new = [l_list[i] for i in ind]

        n_result.append({'train' : type(dataset)(x=x, y=y, classes = already_select_class, target_dtype = dataset.target_dtype),'test':type(dataset)(x=x_new, y=y_new, classes = already_select_class, target_dtype = dataset.target_dtype)})
        n_idx.append(list(already_select_id))

        print("--- LENGTH DATASET: ",len(already_select_id))

    return n_result,n_idx
