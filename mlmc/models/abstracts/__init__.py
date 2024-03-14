from .abstract_textclassification import TextClassificationAbstract
# from .abstract_label import LabelEmbeddingAbstract
from .abstract_encoder import EncoderAbstract
from .abstracts_zeroshot import TextClassificationAbstractZeroShot
from .abstract_embedding import  LabelEmbeddingAbstract

def finetune_lm(representation, data, device="cpu", name=None,epochs=1, batch_size=8, valid=0.1):
    import subprocess, pathlib, os, sys, datetime
    CACHE = pathlib.Path.home() / ".mlmc" / "models"

    if name is None:
        name = f"{str(hash(datetime.datetime.now()))[1:]}"
    if (CACHE/name).exists():
        name = name + "_new"
    out_file = CACHE / name

    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = "" if device == "cpu" else str(device).split(":")[-1]
    cmd = pathlib.Path(__file__).parents[0] / "pretrain-language-model.py"

    subprocess.call(
        [sys.executable, cmd, "--model", representation, "--file", str(data), "--output", str(out_file),
         "--epochs", str(epochs), "--batch_size", str(batch_size), "--valid_frac", str(valid)],
        env=my_env
    )
    return out_file
