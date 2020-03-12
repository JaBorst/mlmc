Module mlmc.save_and_load
=========================
A save and load function for models.

.

Functions
---------

    
`load(path, only_inference=False, only_cpu=False)`
:   Load a model from disk
    
    Args:
        path: The path to load the model from
        only_inference: If the model was saved with only_inference=True this should be True aswell.
        only_cpu: If the model was trained on a GPU, but is trying to be loaded on a system without GPUs set ``only_cpu=True``
            to map the model into RAM instead of GPU.
    
    Returns:
        The loaded model.

    
`save(model, path, only_inference=False)`
:   Saving a model to disk
    
    Args:
        model: The model
        path:  The path to save the model to
        only_inference: If False, the optimizer and loss function are saved to disk aswell.
    
    Returns:
        The path the model was saved to