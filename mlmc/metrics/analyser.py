import pathlib
import json

class History:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        with open(self.path, "r") as f:
            self.history = json.loads(f.read())


        if isinstance(self.history, list):
            new_history = {k:[] if k == "valid" else {"loss":[]} for k in self.history[0].keys()}
            for l in self.history:
                for k in new_history.keys():
                    if k == "train":
                        new_history[k]["loss"].append(l[k]["loss"][0])
                    else:
                        new_history[k].append(l[k][0])

            self.history = new_history


    def get(self, of, m, e=-1):
        if e is not None:
            try:
                r = self.history[of][e][m]
            except:
                r = self.history[of][e]["report"][m]
        else:
            try:
                r = [self.history[of][i][m] for i in range(len(self.history[of]))]
            except:
                r = [self.history[of][i]["report"][m] for i in range(len(self.history[of]))]

        return r

    def plot(self, of, m, p=None):
        r = self.get(of, m, None)
        if p is not None: r = [x[p] for x in r]
        import matplotlib.pyplot as plt

        return plt.plot(r, label=m)

    def plot_labels(self, labels,  p=None):
        import matplotlib.pyplot as plt
        for l in labels:
            self.plot("valid",l,p)
        plt.xlabel("Epochs")
        plt.ylabel(p)
        plt.ylim(0,1)
        plt.plot()
        plt.legend()
#
# import matplotlib.pyplot as plt
# path ="/home/jb/history.json"
# h = History(path)
#
# h.plot_labels(["Energy markets", "Sports"],  "f1-score")
# plt.show()
# r = h.get("valid", "Religion", None)
