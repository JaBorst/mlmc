import collections

def flatten(history):
    def _flatten(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(_flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_lists = [_flatten(x) for x in history["valid"]]
    d = {}
    for l in flat_lists:
        for k,v in l.items():
            if "_support" in k: continue
            k = k.replace("report_","")
            if k not in d: d[k]=[]
            d[k].append(v)
    return d

def flt(history):
    def _flatten(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(_flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    history = [history] if isinstance(history, dict) else history
    flat_lists = [_flatten(x) for x in history]
    d = {}
    for l in flat_lists:
        for k,v in l.items():
            if "_support" in k: continue
            k = k.replace("report_","")
            if k not in d: d[k]=[]
            d[k].append(v)
    return d