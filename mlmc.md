Help on module encoder:

NAME
    encoder

CLASSES
    mlmc.models.abstracts.abstract_encoder.EncoderAbstract(mlmc.models.abstracts.abstract_embedding.LabelEmbeddingAbstract)
        Encoder(mlmc.models.abstracts.abstract_encoder.EncoderAbstract, mlmc.models.abstracts.abstracts_zeroshot.TextClassificationAbstractZeroShot)
    mlmc.models.abstracts.abstracts_zeroshot.TextClassificationAbstractZeroShot(torch.nn.modules.module.Module)
        Encoder(mlmc.models.abstracts.abstract_encoder.EncoderAbstract, mlmc.models.abstracts.abstracts_zeroshot.TextClassificationAbstractZeroShot)
    
    class Encoder(mlmc.models.abstracts.abstract_encoder.EncoderAbstract, mlmc.models.abstracts.abstracts_zeroshot.TextClassificationAbstractZeroShot)
     |  Encoder(*args, **kwargs)
     |  
     |  Trainin a model by entailing text and label into an entailment task. Offers good zeroshot capacities when pretrained
     |  on an NLI task. (you can pretrain (almost) any  transformer model with model.pretrain_snli() or model.pretrain_mnli().
     |  
     |  Method resolution order:
     |      Encoder
     |      mlmc.models.abstracts.abstract_encoder.EncoderAbstract
     |      mlmc.models.abstracts.abstract_embedding.LabelEmbeddingAbstract
     |      mlmc.models.abstracts.abstract_textclassification.TextClassificationAbstract
     |      mlmc.models.abstracts.abstracts_zeroshot.TextClassificationAbstractZeroShot
     |      torch.nn.modules.module.Module
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, *args, **kwargs)
     |      Only there to initialize a projection for binary classification
     |  
     |  bayesian_forward(self, x)
     |  
     |  forward(self, x)
     |      Defines the computation performed at every call.
     |      
     |      Should be overridden by all subclasses.
     |      
     |      .. note::
     |          Although the recipe for forward pass needs to be defined within
     |          this function, one should call the :class:`Module` instance afterwards
     |          instead of this since the former takes care of running the
     |          registered hooks while the latter silently ignores them.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from mlmc.models.abstracts.abstract_encoder.EncoderAbstract:
     |  
     |  transform(self, x, h=None, max_length=400, reshape=False, device=None)
     |      A standard transformation function from text to network input format
     |      
     |      The function looks for the tokenizer attribute. If it doesn't exist the transform function has to
     |      be implemented in the child class
     |      
     |      Args:
     |          x: A list of text
     |      
     |      Returns:
     |          A tensor in the network input format.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from mlmc.models.abstracts.abstract_embedding.LabelEmbeddingAbstract:
     |  
     |  create_labels(self, classes: dict)
     |      Method to change the current target variables
     |      Args:
     |          classes: Dictionary of class mapping like {"label1": 0, "label2":1, ...}
     |      
     |      Returns:
     |  
     |  label_embed(self, x)
     |      Label embedder in this instance uses the same transformation as the input
     |      Args:
     |          x:
     |      
     |      Returns:
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from mlmc.models.abstracts.abstract_textclassification.TextClassificationAbstract:
     |  
     |  abc(self, threshold='max', activation=<built-in method softmax of type object at 0x7ff90b2a4ea0>, loss=<class 'torch.nn.modules.loss.CrossEntropyLoss'>, **kwargs)
     |  
     |  act(self, x)
     |      Applies activation function to output tensor.
     |      
     |      :param x: An input tensor
     |      :return: A tensor
     |  
     |  build(self)
     |      Internal build method.
     |  
     |  contrastive_pretrain(self, d, valid=None, valid_steps=100, batch_size=16, steps=10000)
     |  
     |  embed(self, x)
     |      Method to return input embeddings.
     |      ToDo: Modularize the forward to avoid code copying.
     |      Args:
     |          x: list of input texts
     |      
     |      Returns: a tuple of:
     |          A tensor of embeddings shape (b, e), where b is the number of input texts and e the embedding dimension
     |          A tensor of embeddings shape (l, e), where l is the number of labels and e the embedding dimension
     |  
     |  embed_batch(self, data, batch_size=50)
     |      Predict all labels for a dataset int the mlmc.data.MultilabelDataset format.
     |      
     |      For detailed information on the arcuments see `mlmc.models.TextclassificationAbstract.predict`
     |      
     |      Args:
     |          data: A MultilabelDataset
     |          batch_size: Batch size
     |      
     |      Returns:
     |          A list of labels
     |  
     |  embed_input(self, x)
     |      Using a specified representation (language model or glove vectors) embeds an input tensor.
     |      
     |      :param x: Input tensor
     |      :return: Embedded tensor
     |  
     |  entailment(self, threshold='max', activation=<built-in method softmax of type object at 0x7ff90b2a4ea0>, loss=<class 'torch.nn.modules.loss.CrossEntropyLoss'>, **kwargs)
     |      Helper function to set model into default multi label mode
     |  
     |  evaluate(self, data, batch_size=50, mask=None, metrics=None, _fit=False)
     |      Evaluation, return accuracy and loss and some multilabel measure
     |      
     |      Returns p@1, p@3, p@5, AUC, loss, Accuracy@0.5, Accuracy@mcut, ROC Values, class-wise F1, Precision and Recall.
     |      Args:
     |          data: A MultilabelDataset with the data for evaluation
     |          batch_size: The batch size of the evaluation loop. (Larger is often faster, but it should be small enough
     |          to fit into GPU memory. In general it can be larger than batch_size in training.
     |          return_roc: If True, the return dictionary contains the ROC values.
     |          return_report: If True, the return dictionary will contain a class wise report of F1, Precision and Recall.
     |          metrics: Additional metrics
     |      Returns:
     |          A dictionary with the evaluation measurements.
     |  
     |  evaluate_classes(self, classes_subset=None, **kwargs)
     |      wrapper for evaluation function if you just want to evaluate on subsets of the classes.
     |  
     |  evaluate_word_importance(self, x, plot=False)
     |  
     |  finetune_lm(self, file, epochs=1, batch_size=8, valid=0.1)
     |  
     |  fit(self, train, valid=None, epochs=1, batch_size=16, valid_batch_size=50, patience=-1, tolerance=0.01, return_roc=False, return_report=False, callbacks=None, metrics=None, lr_schedule=None, lr_param={}, log_mlflow=False, valid_prefix='valid')
     |      Training function
     |      
     |      Args:
     |          train: MultilabelDataset used as training data
     |          valid: MultilabelDataset to keep track of generalization
     |          epochs: Number of epochs (times to iterate the train data)
     |          batch_size: Number of instances in one batch.
     |          valid_batch_size: Number of instances in one batch  of validation.
     |          patience: (default -1) Early Stopping Arguments.
     |          Number of epochs to wait for performance improvements before exiting the training loop.
     |          tolerance: (default 1e-2) Early Stopping Arguments.
     |          Minimum improvement of an epoch over the best validation loss so far.
     |      
     |      Returns:
     |          A history dictionary with the loss and the validation evaluation measurements.
     |  
     |  kfold(self, data, validation=0.1, *args, k=10, cb_fn=None, **kwargs)
     |  
     |  ktrain(self, data, test, n, *args, runs=5, cb_fn=<function TextClassificationAbstract.<lambda> at 0x7ff860b9d0d0>, **kwargs)
     |  
     |  log_mlflow(self)
     |  
     |  make_NSP_from_dataset(self, d, n=1000)
     |  
     |  multi(self, threshold='mcut', activation=<built-in method sigmoid of type object at 0x7ff90b2a4ea0>, loss=<class 'torch.nn.modules.loss.BCEWithLogitsLoss'>, **kwargs)
     |      Setting the defaults for multi label mode
     |  
     |  num_params(self)
     |      Count the number of trainable parameters.
     |      
     |      Returns:
     |          The number of trainable parameters
     |  
     |  predict(self, x, h=None, return_scores=False)
     |      Classify sentence string  or a list of strings.
     |      
     |      Args:
     |          x:  A list of the text instances.
     |          return_scores:  If True, the labels are returned with their corresponding confidence scores and prediction mask
     |      Returns:
     |          A list of the labels or a tuple of (labels, scores, mask) if return_scores=True
     |  
     |  predict_aspect_based_sentiment(self, x, aspects, mode='commonaspects')
     |  
     |  predict_batch(self, data, h=None, batch_size=50, return_scores=False)
     |      Predict all labels for a dataset int the mlmc.data.MultilabelDataset format.
     |      
     |      For detailed information on the arcuments see `mlmc.models.TextclassificationAbstract.predict`
     |      
     |      Args:
     |          data: A MultilabelDataset
     |          batch_size: Batch size
     |      
     |      Returns:
     |          A list of labels
     |  
     |  predict_dataset(self, data, batch_size=50, return_scores=False)
     |      Predict all labels for a dataset int the mlmc.data.MultilabelDataset format.
     |      
     |      For detailed information on the arcuments see `mlmc.models.TextclassificationAbstract.predict`
     |      
     |      Args:
     |          data: A MultilabelDataset
     |          batch_size: Batch size
     |      
     |      Returns:
     |          A list of labels
     |  
     |  rebuild(self)
     |      Internal build method.
     |  
     |  reset_memory(self)
     |  
     |  run(self, x)
     |      Transforms textual input into network format and applies activation function.
     |      
     |      :param x: A string or a list of strings
     |      :return: A tensor
     |  
     |  same_text_from_dataset(d, n=1000)
     |  
     |  scores(self, x, h=None)
     |      Returns 2D tensor with length of x and number of labels as shape: (N, L)
     |      Args:
     |          x:
     |      
     |      Returns:
     |  
     |  scores_dataset(self, data, return_truth=False, batch_size=50)
     |      Predict all labels for a dataset int the mlmc.data.MultilabelDataset format.
     |      
     |      For detailed information on the arcuments see `mlmc.models.TextclassificationAbstract.predict`
     |      
     |      Args:
     |          data: A MultilabelDataset
     |          batch_size: Batch size
     |      
     |      Returns:
     |          A list of labels
     |  
     |  set_activation(self, name)
     |  
     |  set_device(self, device)
     |  
     |  set_loss(self, loss)
     |  
     |  set_optimizer(self, optimizer, optimizer_params={})
     |  
     |  set_sformatter(self, c)
     |      Setter for the label sformatter
     |      Args:
     |          c: callable that takes and returns a string
     |      
     |      Returns:
     |  
     |  set_threshold(self, name)
     |      Sets the threshold function which will be used to as a decision threshold.
     |      
     |      :param name: Name of the threshold (see mlmc.thresholds.threshold_dict.keys())
     |  
     |  single(self, threshold='max', activation=<built-in method softmax of type object at 0x7ff90b2a4ea0>, loss=<class 'torch.nn.modules.loss.CrossEntropyLoss'>, **kwargs)
     |      Setting the default single label mode
     |  
     |  sts(self, threshold='mcut', activation=<function TextClassificationAbstract.<lambda> at 0x7ff860b9dc10>, loss=<class 'mlmc.loss.loss_labelwise_ranking.RelativeRankingLoss'>, **kwargs)
     |      Helper function to set model into default sts mode
     |  
     |  tm_pretrain(self, d, valid=None, valid_steps=100, batch_size=16, steps=10000)
     |  
     |  zero_contrastive_pretrain(self, d, valid=None, valid_steps=100, batch_size=16, valid_batch_size=16, steps=10000, log_mlflow=False)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from mlmc.models.abstracts.abstracts_zeroshot.TextClassificationAbstractZeroShot:
     |  
     |  pretrain_entailment(self, train, valid=None, steps=10000, eval_every=1000, datasets=None, formatters=None, batch_size=16, valid_batch_size=32, callbacks=None, lr_schedule=None, lr_param={}, log_mlflow=False, sample_size=-1)
     |      Training function
     |      
     |      Args:
     |          train: MultilabelDataset used as training data
     |          valid: MultilabelDataset to keep track of generalization
     |          epochs: Number of epochs (times to iterate the train data)
     |          batch_size: Number of instances in one batch.
     |          valid_batch_size: Number of instances in one batch  of validation.
     |          patience: (default -1) Early Stopping Arguments.
     |          Number of epochs to wait for performance improvements before exiting the training loop.
     |          tolerance: (default 1e-2) Early Stopping Arguments.
     |          Minimum improvement of an epoch over the best validation loss so far.
     |      
     |      Returns:
     |          A history dictionary with the loss and the validation evaluation measurements.
     |  
     |  pretrain_sts(self, batch_size=12, datasets=None, steps=600, eval_every=100, log_mlflow=False)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from torch.nn.modules.module.Module:
     |  
     |  __call__ = _call_impl(self, *input, **kwargs)
     |  
     |  __delattr__(self, name)
     |      Implement delattr(self, name).
     |  
     |  __dir__(self)
     |      Default dir() implementation.
     |  
     |  __getattr__(self, name: str) -> Union[torch.Tensor, ForwardRef('Module')]
     |  
     |  __repr__(self)
     |      Return repr(self).
     |  
     |  __setattr__(self, name: str, value: Union[torch.Tensor, ForwardRef('Module')]) -> None
     |      Implement setattr(self, name, value).
     |  
     |  __setstate__(self, state)
     |  
     |  add_module(self, name: str, module: Union[ForwardRef('Module'), NoneType]) -> None
     |      Adds a child module to the current module.
     |      
     |      The module can be accessed as an attribute using the given name.
     |      
     |      Args:
     |          name (string): name of the child module. The child module can be
     |              accessed from this module using the given name
     |          module (Module): child module to be added to the module.
     |  
     |  apply(self: ~T, fn: Callable[[ForwardRef('Module')], NoneType]) -> ~T
     |      Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
     |      as well as self. Typical use includes initializing the parameters of a model
     |      (see also :ref:`nn-init-doc`).
     |      
     |      Args:
     |          fn (:class:`Module` -> None): function to be applied to each submodule
     |      
     |      Returns:
     |          Module: self
     |      
     |      Example::
     |      
     |          >>> @torch.no_grad()
     |          >>> def init_weights(m):
     |          >>>     print(m)
     |          >>>     if type(m) == nn.Linear:
     |          >>>         m.weight.fill_(1.0)
     |          >>>         print(m.weight)
     |          >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
     |          >>> net.apply(init_weights)
     |          Linear(in_features=2, out_features=2, bias=True)
     |          Parameter containing:
     |          tensor([[ 1.,  1.],
     |                  [ 1.,  1.]])
     |          Linear(in_features=2, out_features=2, bias=True)
     |          Parameter containing:
     |          tensor([[ 1.,  1.],
     |                  [ 1.,  1.]])
     |          Sequential(
     |            (0): Linear(in_features=2, out_features=2, bias=True)
     |            (1): Linear(in_features=2, out_features=2, bias=True)
     |          )
     |          Sequential(
     |            (0): Linear(in_features=2, out_features=2, bias=True)
     |            (1): Linear(in_features=2, out_features=2, bias=True)
     |          )
     |  
     |  bfloat16(self: ~T) -> ~T
     |      Casts all floating point parameters and buffers to ``bfloat16`` datatype.
     |      
     |      .. note::
     |          This method modifies the module in-place.
     |      
     |      Returns:
     |          Module: self
     |  
     |  buffers(self, recurse: bool = True) -> Iterator[torch.Tensor]
     |      Returns an iterator over module buffers.
     |      
     |      Args:
     |          recurse (bool): if True, then yields buffers of this module
     |              and all submodules. Otherwise, yields only buffers that
     |              are direct members of this module.
     |      
     |      Yields:
     |          torch.Tensor: module buffer
     |      
     |      Example::
     |      
     |          >>> for buf in model.buffers():
     |          >>>     print(type(buf), buf.size())
     |          <class 'torch.Tensor'> (20L,)
     |          <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
     |  
     |  children(self) -> Iterator[ForwardRef('Module')]
     |      Returns an iterator over immediate children modules.
     |      
     |      Yields:
     |          Module: a child module
     |  
     |  cpu(self: ~T) -> ~T
     |      Moves all model parameters and buffers to the CPU.
     |      
     |      .. note::
     |          This method modifies the module in-place.
     |      
     |      Returns:
     |          Module: self
     |  
     |  cuda(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T
     |      Moves all model parameters and buffers to the GPU.
     |      
     |      This also makes associated parameters and buffers different objects. So
     |      it should be called before constructing optimizer if the module will
     |      live on GPU while being optimized.
     |      
     |      .. note::
     |          This method modifies the module in-place.
     |      
     |      Args:
     |          device (int, optional): if specified, all parameters will be
     |              copied to that device
     |      
     |      Returns:
     |          Module: self
     |  
     |  double(self: ~T) -> ~T
     |      Casts all floating point parameters and buffers to ``double`` datatype.
     |      
     |      .. note::
     |          This method modifies the module in-place.
     |      
     |      Returns:
     |          Module: self
     |  
     |  eval(self: ~T) -> ~T
     |      Sets the module in evaluation mode.
     |      
     |      This has any effect only on certain modules. See documentations of
     |      particular modules for details of their behaviors in training/evaluation
     |      mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
     |      etc.
     |      
     |      This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.
     |      
     |      See :ref:`locally-disable-grad-doc` for a comparison between
     |      `.eval()` and several similar mechanisms that may be confused with it.
     |      
     |      Returns:
     |          Module: self
     |  
     |  extra_repr(self) -> str
     |      Set the extra representation of the module
     |      
     |      To print customized extra information, you should re-implement
     |      this method in your own modules. Both single-line and multi-line
     |      strings are acceptable.
     |  
     |  float(self: ~T) -> ~T
     |      Casts all floating point parameters and buffers to ``float`` datatype.
     |      
     |      .. note::
     |          This method modifies the module in-place.
     |      
     |      Returns:
     |          Module: self
     |  
     |  get_buffer(self, target: str) -> 'Tensor'
     |      Returns the buffer given by ``target`` if it exists,
     |      otherwise throws an error.
     |      
     |      See the docstring for ``get_submodule`` for a more detailed
     |      explanation of this method's functionality as well as how to
     |      correctly specify ``target``.
     |      
     |      Args:
     |          target: The fully-qualified string name of the buffer
     |              to look for. (See ``get_submodule`` for how to specify a
     |              fully-qualified string.)
     |      
     |      Returns:
     |          torch.Tensor: The buffer referenced by ``target``
     |      
     |      Raises:
     |          AttributeError: If the target string references an invalid
     |              path or resolves to something that is not a
     |              buffer
     |  
     |  get_parameter(self, target: str) -> 'Parameter'
     |      Returns the parameter given by ``target`` if it exists,
     |      otherwise throws an error.
     |      
     |      See the docstring for ``get_submodule`` for a more detailed
     |      explanation of this method's functionality as well as how to
     |      correctly specify ``target``.
     |      
     |      Args:
     |          target: The fully-qualified string name of the Parameter
     |              to look for. (See ``get_submodule`` for how to specify a
     |              fully-qualified string.)
     |      
     |      Returns:
     |          torch.nn.Parameter: The Parameter referenced by ``target``
     |      
     |      Raises:
     |          AttributeError: If the target string references an invalid
     |              path or resolves to something that is not an
     |              ``nn.Parameter``
     |  
     |  get_submodule(self, target: str) -> 'Module'
     |      Returns the submodule given by ``target`` if it exists,
     |      otherwise throws an error.
     |      
     |      For example, let's say you have an ``nn.Module`` ``A`` that
     |      looks like this:
     |      
     |      .. code-block::text
     |      
     |          A(
     |              (net_b): Module(
     |                  (net_c): Module(
     |                      (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
     |                  )
     |                  (linear): Linear(in_features=100, out_features=200, bias=True)
     |              )
     |          )
     |      
     |      (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
     |      submodule ``net_b``, which itself has two submodules ``net_c``
     |      and ``linear``. ``net_c`` then has a submodule ``conv``.)
     |      
     |      To check whether or not we have the ``linear`` submodule, we
     |      would call ``get_submodule("net_b.linear")``. To check whether
     |      we have the ``conv`` submodule, we would call
     |      ``get_submodule("net_b.net_c.conv")``.
     |      
     |      The runtime of ``get_submodule`` is bounded by the degree
     |      of module nesting in ``target``. A query against
     |      ``named_modules`` achieves the same result, but it is O(N) in
     |      the number of transitive modules. So, for a simple check to see
     |      if some submodule exists, ``get_submodule`` should always be
     |      used.
     |      
     |      Args:
     |          target: The fully-qualified string name of the submodule
     |              to look for. (See above example for how to specify a
     |              fully-qualified string.)
     |      
     |      Returns:
     |          torch.nn.Module: The submodule referenced by ``target``
     |      
     |      Raises:
     |          AttributeError: If the target string references an invalid
     |              path or resolves to something that is not an
     |              ``nn.Module``
     |  
     |  half(self: ~T) -> ~T
     |      Casts all floating point parameters and buffers to ``half`` datatype.
     |      
     |      .. note::
     |          This method modifies the module in-place.
     |      
     |      Returns:
     |          Module: self
     |  
     |  load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True)
     |      Copies parameters and buffers from :attr:`state_dict` into
     |      this module and its descendants. If :attr:`strict` is ``True``, then
     |      the keys of :attr:`state_dict` must exactly match the keys returned
     |      by this module's :meth:`~torch.nn.Module.state_dict` function.
     |      
     |      Args:
     |          state_dict (dict): a dict containing parameters and
     |              persistent buffers.
     |          strict (bool, optional): whether to strictly enforce that the keys
     |              in :attr:`state_dict` match the keys returned by this module's
     |              :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
     |      
     |      Returns:
     |          ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
     |              * **missing_keys** is a list of str containing the missing keys
     |              * **unexpected_keys** is a list of str containing the unexpected keys
     |  
     |  modules(self) -> Iterator[ForwardRef('Module')]
     |      Returns an iterator over all modules in the network.
     |      
     |      Yields:
     |          Module: a module in the network
     |      
     |      Note:
     |          Duplicate modules are returned only once. In the following
     |          example, ``l`` will be returned only once.
     |      
     |      Example::
     |      
     |          >>> l = nn.Linear(2, 2)
     |          >>> net = nn.Sequential(l, l)
     |          >>> for idx, m in enumerate(net.modules()):
     |                  print(idx, '->', m)
     |      
     |          0 -> Sequential(
     |            (0): Linear(in_features=2, out_features=2, bias=True)
     |            (1): Linear(in_features=2, out_features=2, bias=True)
     |          )
     |          1 -> Linear(in_features=2, out_features=2, bias=True)
     |  
     |  named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]
     |      Returns an iterator over module buffers, yielding both the
     |      name of the buffer as well as the buffer itself.
     |      
     |      Args:
     |          prefix (str): prefix to prepend to all buffer names.
     |          recurse (bool): if True, then yields buffers of this module
     |              and all submodules. Otherwise, yields only buffers that
     |              are direct members of this module.
     |      
     |      Yields:
     |          (string, torch.Tensor): Tuple containing the name and buffer
     |      
     |      Example::
     |      
     |          >>> for name, buf in self.named_buffers():
     |          >>>    if name in ['running_var']:
     |          >>>        print(buf.size())
     |  
     |  named_children(self) -> Iterator[Tuple[str, ForwardRef('Module')]]
     |      Returns an iterator over immediate children modules, yielding both
     |      the name of the module as well as the module itself.
     |      
     |      Yields:
     |          (string, Module): Tuple containing a name and child module
     |      
     |      Example::
     |      
     |          >>> for name, module in model.named_children():
     |          >>>     if name in ['conv4', 'conv5']:
     |          >>>         print(module)
     |  
     |  named_modules(self, memo: Union[Set[ForwardRef('Module')], NoneType] = None, prefix: str = '', remove_duplicate: bool = True)
     |      Returns an iterator over all modules in the network, yielding
     |      both the name of the module as well as the module itself.
     |      
     |      Args:
     |          memo: a memo to store the set of modules already added to the result
     |          prefix: a prefix that will be added to the name of the module
     |          remove_duplicate: whether to remove the duplicated module instances in the result
     |          or not
     |      
     |      Yields:
     |          (string, Module): Tuple of name and module
     |      
     |      Note:
     |          Duplicate modules are returned only once. In the following
     |          example, ``l`` will be returned only once.
     |      
     |      Example::
     |      
     |          >>> l = nn.Linear(2, 2)
     |          >>> net = nn.Sequential(l, l)
     |          >>> for idx, m in enumerate(net.named_modules()):
     |                  print(idx, '->', m)
     |      
     |          0 -> ('', Sequential(
     |            (0): Linear(in_features=2, out_features=2, bias=True)
     |            (1): Linear(in_features=2, out_features=2, bias=True)
     |          ))
     |          1 -> ('0', Linear(in_features=2, out_features=2, bias=True))
     |  
     |  named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.nn.parameter.Parameter]]
     |      Returns an iterator over module parameters, yielding both the
     |      name of the parameter as well as the parameter itself.
     |      
     |      Args:
     |          prefix (str): prefix to prepend to all parameter names.
     |          recurse (bool): if True, then yields parameters of this module
     |              and all submodules. Otherwise, yields only parameters that
     |              are direct members of this module.
     |      
     |      Yields:
     |          (string, Parameter): Tuple containing the name and parameter
     |      
     |      Example::
     |      
     |          >>> for name, param in self.named_parameters():
     |          >>>    if name in ['bias']:
     |          >>>        print(param.size())
     |  
     |  parameters(self, recurse: bool = True) -> Iterator[torch.nn.parameter.Parameter]
     |      Returns an iterator over module parameters.
     |      
     |      This is typically passed to an optimizer.
     |      
     |      Args:
     |          recurse (bool): if True, then yields parameters of this module
     |              and all submodules. Otherwise, yields only parameters that
     |              are direct members of this module.
     |      
     |      Yields:
     |          Parameter: module parameter
     |      
     |      Example::
     |      
     |          >>> for param in model.parameters():
     |          >>>     print(type(param), param.size())
     |          <class 'torch.Tensor'> (20L,)
     |          <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
     |  
     |  register_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, torch.Tensor]]) -> torch.utils.hooks.RemovableHandle
     |      Registers a backward hook on the module.
     |      
     |      This function is deprecated in favor of :meth:`nn.Module.register_full_backward_hook` and
     |      the behavior of this function will change in future versions.
     |      
     |      Returns:
     |          :class:`torch.utils.hooks.RemovableHandle`:
     |              a handle that can be used to remove the added hook by calling
     |              ``handle.remove()``
     |  
     |  register_buffer(self, name: str, tensor: Union[torch.Tensor, NoneType], persistent: bool = True) -> None
     |      Adds a buffer to the module.
     |      
     |      This is typically used to register a buffer that should not to be
     |      considered a model parameter. For example, BatchNorm's ``running_mean``
     |      is not a parameter, but is part of the module's state. Buffers, by
     |      default, are persistent and will be saved alongside parameters. This
     |      behavior can be changed by setting :attr:`persistent` to ``False``. The
     |      only difference between a persistent buffer and a non-persistent buffer
     |      is that the latter will not be a part of this module's
     |      :attr:`state_dict`.
     |      
     |      Buffers can be accessed as attributes using given names.
     |      
     |      Args:
     |          name (string): name of the buffer. The buffer can be accessed
     |              from this module using the given name
     |          tensor (Tensor): buffer to be registered.
     |          persistent (bool): whether the buffer is part of this module's
     |              :attr:`state_dict`.
     |      
     |      Example::
     |      
     |          >>> self.register_buffer('running_mean', torch.zeros(num_features))
     |  
     |  register_forward_hook(self, hook: Callable[..., NoneType]) -> torch.utils.hooks.RemovableHandle
     |      Registers a forward hook on the module.
     |      
     |      The hook will be called every time after :func:`forward` has computed an output.
     |      It should have the following signature::
     |      
     |          hook(module, input, output) -> None or modified output
     |      
     |      The input contains only the positional arguments given to the module.
     |      Keyword arguments won't be passed to the hooks and only to the ``forward``.
     |      The hook can modify the output. It can modify the input inplace but
     |      it will not have effect on forward since this is called after
     |      :func:`forward` is called.
     |      
     |      Returns:
     |          :class:`torch.utils.hooks.RemovableHandle`:
     |              a handle that can be used to remove the added hook by calling
     |              ``handle.remove()``
     |  
     |  register_forward_pre_hook(self, hook: Callable[..., NoneType]) -> torch.utils.hooks.RemovableHandle
     |      Registers a forward pre-hook on the module.
     |      
     |      The hook will be called every time before :func:`forward` is invoked.
     |      It should have the following signature::
     |      
     |          hook(module, input) -> None or modified input
     |      
     |      The input contains only the positional arguments given to the module.
     |      Keyword arguments won't be passed to the hooks and only to the ``forward``.
     |      The hook can modify the input. User can either return a tuple or a
     |      single modified value in the hook. We will wrap the value into a tuple
     |      if a single value is returned(unless that value is already a tuple).
     |      
     |      Returns:
     |          :class:`torch.utils.hooks.RemovableHandle`:
     |              a handle that can be used to remove the added hook by calling
     |              ``handle.remove()``
     |  
     |  register_full_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, torch.Tensor]]) -> torch.utils.hooks.RemovableHandle
     |      Registers a backward hook on the module.
     |      
     |      The hook will be called every time the gradients with respect to module
     |      inputs are computed. The hook should have the following signature::
     |      
     |          hook(module, grad_input, grad_output) -> tuple(Tensor) or None
     |      
     |      The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
     |      with respect to the inputs and outputs respectively. The hook should
     |      not modify its arguments, but it can optionally return a new gradient with
     |      respect to the input that will be used in place of :attr:`grad_input` in
     |      subsequent computations. :attr:`grad_input` will only correspond to the inputs given
     |      as positional arguments and all kwarg arguments are ignored. Entries
     |      in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
     |      arguments.
     |      
     |      .. warning ::
     |          Modifying inputs or outputs inplace is not allowed when using backward hooks and
     |          will raise an error.
     |      
     |      Returns:
     |          :class:`torch.utils.hooks.RemovableHandle`:
     |              a handle that can be used to remove the added hook by calling
     |              ``handle.remove()``
     |  
     |  register_parameter(self, name: str, param: Union[torch.nn.parameter.Parameter, NoneType]) -> None
     |      Adds a parameter to the module.
     |      
     |      The parameter can be accessed as an attribute using given name.
     |      
     |      Args:
     |          name (string): name of the parameter. The parameter can be accessed
     |              from this module using the given name
     |          param (Parameter): parameter to be added to the module.
     |  
     |  requires_grad_(self: ~T, requires_grad: bool = True) -> ~T
     |      Change if autograd should record operations on parameters in this
     |      module.
     |      
     |      This method sets the parameters' :attr:`requires_grad` attributes
     |      in-place.
     |      
     |      This method is helpful for freezing part of the module for finetuning
     |      or training parts of a model individually (e.g., GAN training).
     |      
     |      See :ref:`locally-disable-grad-doc` for a comparison between
     |      `.requires_grad_()` and several similar mechanisms that may be confused with it.
     |      
     |      Args:
     |          requires_grad (bool): whether autograd should record operations on
     |                                parameters in this module. Default: ``True``.
     |      
     |      Returns:
     |          Module: self
     |  
     |  share_memory(self: ~T) -> ~T
     |      See :meth:`torch.Tensor.share_memory_`
     |  
     |  state_dict(self, destination=None, prefix='', keep_vars=False)
     |      Returns a dictionary containing a whole state of the module.
     |      
     |      Both parameters and persistent buffers (e.g. running averages) are
     |      included. Keys are corresponding parameter and buffer names.
     |      
     |      Returns:
     |          dict:
     |              a dictionary containing a whole state of the module
     |      
     |      Example::
     |      
     |          >>> module.state_dict().keys()
     |          ['bias', 'weight']
     |  
     |  to(self, *args, **kwargs)
     |      Moves and/or casts the parameters and buffers.
     |      
     |      This can be called as
     |      
     |      .. function:: to(device=None, dtype=None, non_blocking=False)
     |      
     |      .. function:: to(dtype, non_blocking=False)
     |      
     |      .. function:: to(tensor, non_blocking=False)
     |      
     |      .. function:: to(memory_format=torch.channels_last)
     |      
     |      Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
     |      floating point or complex :attr:`dtype`s. In addition, this method will
     |      only cast the floating point or complex parameters and buffers to :attr:`dtype`
     |      (if given). The integral parameters and buffers will be moved
     |      :attr:`device`, if that is given, but with dtypes unchanged. When
     |      :attr:`non_blocking` is set, it tries to convert/move asynchronously
     |      with respect to the host if possible, e.g., moving CPU Tensors with
     |      pinned memory to CUDA devices.
     |      
     |      See below for examples.
     |      
     |      .. note::
     |          This method modifies the module in-place.
     |      
     |      Args:
     |          device (:class:`torch.device`): the desired device of the parameters
     |              and buffers in this module
     |          dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
     |              the parameters and buffers in this module
     |          tensor (torch.Tensor): Tensor whose dtype and device are the desired
     |              dtype and device for all parameters and buffers in this module
     |          memory_format (:class:`torch.memory_format`): the desired memory
     |              format for 4D parameters and buffers in this module (keyword
     |              only argument)
     |      
     |      Returns:
     |          Module: self
     |      
     |      Examples::
     |      
     |          >>> linear = nn.Linear(2, 2)
     |          >>> linear.weight
     |          Parameter containing:
     |          tensor([[ 0.1913, -0.3420],
     |                  [-0.5113, -0.2325]])
     |          >>> linear.to(torch.double)
     |          Linear(in_features=2, out_features=2, bias=True)
     |          >>> linear.weight
     |          Parameter containing:
     |          tensor([[ 0.1913, -0.3420],
     |                  [-0.5113, -0.2325]], dtype=torch.float64)
     |          >>> gpu1 = torch.device("cuda:1")
     |          >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
     |          Linear(in_features=2, out_features=2, bias=True)
     |          >>> linear.weight
     |          Parameter containing:
     |          tensor([[ 0.1914, -0.3420],
     |                  [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
     |          >>> cpu = torch.device("cpu")
     |          >>> linear.to(cpu)
     |          Linear(in_features=2, out_features=2, bias=True)
     |          >>> linear.weight
     |          Parameter containing:
     |          tensor([[ 0.1914, -0.3420],
     |                  [-0.5112, -0.2324]], dtype=torch.float16)
     |      
     |          >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
     |          >>> linear.weight
     |          Parameter containing:
     |          tensor([[ 0.3741+0.j,  0.2382+0.j],
     |                  [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
     |          >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
     |          tensor([[0.6122+0.j, 0.1150+0.j],
     |                  [0.6122+0.j, 0.1150+0.j],
     |                  [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)
     |  
     |  to_empty(self: ~T, *, device: Union[str, torch.device]) -> ~T
     |      Moves the parameters and buffers to the specified device without copying storage.
     |      
     |      Args:
     |          device (:class:`torch.device`): The desired device of the parameters
     |              and buffers in this module.
     |      
     |      Returns:
     |          Module: self
     |  
     |  train(self: ~T, mode: bool = True) -> ~T
     |      Sets the module in training mode.
     |      
     |      This has any effect only on certain modules. See documentations of
     |      particular modules for details of their behaviors in training/evaluation
     |      mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
     |      etc.
     |      
     |      Args:
     |          mode (bool): whether to set training mode (``True``) or evaluation
     |                       mode (``False``). Default: ``True``.
     |      
     |      Returns:
     |          Module: self
     |  
     |  type(self: ~T, dst_type: Union[torch.dtype, str]) -> ~T
     |      Casts all parameters and buffers to :attr:`dst_type`.
     |      
     |      .. note::
     |          This method modifies the module in-place.
     |      
     |      Args:
     |          dst_type (type or string): the desired type
     |      
     |      Returns:
     |          Module: self
     |  
     |  xpu(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T
     |      Moves all model parameters and buffers to the XPU.
     |      
     |      This also makes associated parameters and buffers different objects. So
     |      it should be called before constructing optimizer if the module will
     |      live on XPU while being optimized.
     |      
     |      .. note::
     |          This method modifies the module in-place.
     |      
     |      Arguments:
     |          device (int, optional): if specified, all parameters will be
     |              copied to that device
     |      
     |      Returns:
     |          Module: self
     |  
     |  zero_grad(self, set_to_none: bool = False) -> None
     |      Sets gradients of all model parameters to zero. See similar function
     |      under :class:`torch.optim.Optimizer` for more context.
     |      
     |      Args:
     |          set_to_none (bool): instead of setting to zero, set the grads to None.
     |              See :meth:`torch.optim.Optimizer.zero_grad` for details.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from torch.nn.modules.module.Module:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from torch.nn.modules.module.Module:
     |  
     |  T_destination = ~T_destination
     |  
     |  __annotations__ = {'__call__': typing.Callable[..., typing.Any], '_is_...
     |  
     |  dump_patches = False

FILE
    /home/jb/git/mlmc/mlmc/models/zeroshot/encoder.py


