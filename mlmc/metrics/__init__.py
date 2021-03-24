from .multilabel import MultiLabelReport, AUC_ROC
from .precisionk import PrecisionK, AccuracyTreshold
from .analyser import History
from .confusion import ConfusionMatrix

from ..thresholds import get as thresholdget
from .save_scores import SaveScores
from .helpers import flatten, flt

metrics_dict= {
    "p@1": lambda: PrecisionK(k=1, is_multilabel=True, average=True),
    "p@3": lambda: PrecisionK(k=3, is_multilabel=True, average=True),
    "p@5": lambda: PrecisionK(k=5, is_multilabel=True, average=True),
    "tr@0.5": lambda: AccuracyTreshold(trf=thresholdget("hard", 0.5), is_multilabel=True),
    "mcut": lambda: AccuracyTreshold(trf=thresholdget("mcut"), is_multilabel=True),
    "auc_roc":lambda: AUC_ROC(return_roc=True),
    "multilabel_report": lambda:  MultiLabelReport(),
    "accuracy": lambda: AccuracyTreshold(thresholdget("max"), is_multilabel=False),
    "singlelabel_report": lambda: MultiLabelReport(is_multilabel=False)
}

metrics_config = {
    "default_multilabel": ["p@1", "p@3", "p@5", "tr@0.5", "mcut", "auc_roc", "multilabel_report"],
    "default_singlelabel": ["accuracy", "singlelabel_report"]
}

def get(s) -> dict:
    """
    Function for instantiating metrics.

    :param s: Metric name(s), see metrics_dict.keys() and metrics_config.keys() for available names
    :return: Dictionary of form {"metric_name": initialized_metric}
    """
    initial_list = [s] if isinstance(s, str) else s
    remove = []
    obj = []
    for e in initial_list:
        if e in metrics_config.keys():
            initial_list.extend(metrics_config[e])
            remove.append(e)
        if not isinstance(e, str):
            obj.append(e)
    if len(remove) != 0:
        for r in remove:
            initial_list.remove(r)

    r = {x: metrics_dict[x]() if isinstance(x,str) else x for x in initial_list}
    return {k if isinstance(k, str) else v.__class__.__name__: v for k, v in r.items()}


class MetricsDict:
    def __init__(self, map=None):
        """
        Initializes class with metrics.

        :param map: Metric name(s), see metrics_dict.keys() and metrics_config.keys() for available names
        """
        if isinstance(map, dict):
            self.map = map
        else:
            self.map = get(map)
        self.PRECISION_DIGITS = 8

    def init(self, args_dict):
        """
        This function gets to look into the current model the metric is evaluated on an see the class dict.
        This is meant to set variables to the current model
        """
        for v in self.map.values():
            if hasattr(v, "init"):
                v.init(**args_dict)
        for v in self.map.values():
            v.reset()

    def __getitem__(self, item):
        """
        Retrieves a dictionary entry.

        :param item: Key
        :return: Value
        """
        return self.map[item]

    def __iter__(self):
        """Returns iterator object over metric keys"""
        return self.map.keys()

    def values(self):
        """Returns all metric values"""
        return self.map.values()

    def keys(self):
        """Returns all metric keys"""
        return self.map.keys()

    def items(self):
        """Returns all key/value pairs"""
        return self.map.items()

    def update(self, d):
        """
        Adds a key/value pair to the dictionary.

        :param d: A key/value pair to be added
        """
        self.map.update(d)

    def reset(self):
        """Clears all instance attributes of the metrics"""
        for v in self.values():
            v.reset()

    def update_metrics(self, batch):
        """Adds output of classification task in form (scores, truth, pred) to metrics"""
        for v in self.values():
            v.update(batch)

    def compute(self):
        """Computes and returns metric in a dictionary with the metric name as key and metric results as value"""
        r = {k: v.compute() if not isinstance(v, float) else v for k, v in self.map.items()}
        r = {k: round(v, self.PRECISION_DIGITS) if isinstance(v, float) else v for k,v in r.items()}
        return r

    def print(self):
        """Computes and returns metric in a dictionary with the metric name as key and metric results as value by usage
        of print() if it's implemented for the given metric"""
        def _choose(v):
            """
            Chooses a computation function.

            :param v: An instantiated metric
            :return: Function call of print() if it exists else compute()
            """
            if hasattr(v, "print"):
                return v.print()
            if hasattr(v, "compute"):
                return v.compute()
            else:
                return v
        r = {k: _choose(v) for k, v in self.map.items()}
        return r

    def _recurse_dictionary(self, d, prefix=""):
        """
        Recurses through dictionary of metric results and adds them to a list.

        :param d: Dictionary containing the results of the chosen metrics
        :param prefix: Prefix added to metric
        :return: List of metric results
        """
        l = []
        for k, v in d.items():
            if isinstance(v, float):
                l.append((f"{prefix}_{k}", v))
            elif isinstance(v, dict):
                l.extend(self._recurse_dictionary(v, "_".join([prefix, k])))
            elif isinstance(v, (list, tuple)):
                for e,i in enumerate(v):
                    if isinstance(i,float):
                        l.append((f"{prefix}_{k}_{e}", i))
                    elif isinstance(i, dict):
                        l.extend(self._recurse_dictionary(i, "_".join([prefix, k])))
                    else:
                        print("This is a list of floats")
        return l

    def log_sacred(self, _run, step, prefix=""):
        """
        Logs a metric to Sacred.

        :param _run: Run object of Experiment
        :param step: Iteration step in which the metric was taken
        :param prefix: Prefix added to metric
        :return: Run object of Experiment with added metric
        """
        results = self.print()
        for k, v in self._recurse_dictionary(results, prefix=prefix):
            _run.log_scalar(k,v,step)
        return _run

    def log_mlflow(self, _run, step, prefix=""):
        """
        Logs a metric to MLflow.

        :param _run: Run object of Experiment
        :param step: Iteration step in which the metric was taken
        :param prefix: Prefix added to metric
        :return: Run object of Experiment with added metric
        """
        import mlflow
        results = self.print()
        for k, v in self._recurse_dictionary(results, prefix=prefix):
            mlflow.log_metric(k.replace("@","/a/"),v,step)
        return _run

    def __repr__(self):
        return ", ".join(self.map.keys())

    def rename(self, map: dict):
        self.map = {map.get(k,k):v for k,v in self.map.items()}