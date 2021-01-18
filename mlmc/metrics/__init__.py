from .multilabel import MultiLabelReport, AUC_ROC
from .precisionk import PrecisionK, AccuracyTreshold
from .analyser import History
from .confusion import ConfusionMatrix

from ..thresholds import get as thresholdget
from .save_scores import SaveScores

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
        return self.map[item]

    def __iter__(self):
        return self.map.keys()

    def values(self):
        return self.map.values()

    def keys(self):
        return self.map.keys()

    def items(self):
        return self.map.items()

    def update(self, d):
        self.map.update(d)

    def reset(self):
        for v in self.values():
            v.reset()

    def update_metrics(self, batch):
        for v in self.values():
            v.update(batch)

    def compute(self):
        r = {k: v.compute() if not isinstance(v, float) else v for k, v in self.map.items()}
        r = {k: round(v, self.PRECISION_DIGITS) if isinstance(v, float) else v for k,v in r.items()}
        return r

    def print(self):
        def _choose(v):
            if hasattr(v, "print"):
                return v.print()
            if hasattr(v, "compute"):
                return v.compute()
            else:
                return v
        r = {k: _choose(v) for k, v in self.map.items()}
        return r

    def _recurse_dictionary(self, d, prefix=""):
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
        results = self.print()
        for k, v in self._recurse_dictionary(results, prefix=prefix):
            _run.log_scalar(k,v,step)
        return _run

    def log_mlflow(self, _run, step, prefix=""):
        import mlflow
        results = self.print()
        for k, v in self._recurse_dictionary(results, prefix=prefix):
            mlflow.log_metric(k,v,step)
        return _run

    def __repr__(self):
        return ", ".join(self.map.keys())
