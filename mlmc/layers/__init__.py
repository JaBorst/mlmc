from .probability import Prob
from .lstm import LSTM, LSTMRD
from .label_layers import LabelEmbeddingScoring,AdaptiveCombination,LabelAttention, LabelSpecificSelfAttention
from .metric_layers import Bilinear, Metric
from .weighted_aggregation import AttentionWeightedAggregation
from .mogrifier import MogrifierLSTM
# Make pytorch_geometric effectively optional
try:
    from .graph import GatedGraphConv
except:
    print("pytorch_geometric not installed. Install if you want to use Geometric models")