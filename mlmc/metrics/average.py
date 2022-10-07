from ignite.metrics import Average
# class Average():
#     """Averaging"""
#     def __init__(self,):
#         self.reset()
#
#     def init(self, classes, **kwargs):
#         "an extra function for model specific parameters of the metric"
#         self.classes = classes
#
#     def reset(self):
#         """Clears previously added truth and pred instance attributes."""
#         self.n = 0
#         self.val = []
#
#     def update(self, batch):
#         """
#         Adds classification output to class for computation of metric.
#
#         :param batch: Output of classification task in form (scores, truth, pred)
#         """
#         self.n = self.n + 1
#         if isinstance(batch, float):
#             self.val.append(batch)
#         elif isinstance(batch, list):
#             self.val.extend(batch)
#         else:
#             self.val = self.val + list(batch)
#
#     def compute(self,*args, **kwargs):
#         """
#         Computes metric.
#
#         :return: average of values
#         """
#         return sum(self.val) / self.n


