from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, LineByLineTextDataset
import argparse
import pathlib
from transformers import Trainer, TrainingArguments
import numpy as np
from copy import deepcopy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Basic Version of finetuning a language on domain specific data.')
    parser.add_argument('--model', metavar='model', type=str, help='Model name')
    parser.add_argument('--file', dest='file', help='Raw text input file')
    parser.add_argument('--output', dest='output',  help='Raw text input file')
    parser.add_argument('--epochs', dest='epochs', type=int,help='Raw text input file')
    parser.add_argument('--batch_size', dest='batch_size', type=int,help='Raw text input file')
    parser.add_argument('--valid_fraction', dest='valid', type=float,help='fraction of data to be used as validation data')
    args = parser.parse_args()

    repr=args.model
    print("Language Model Pretraining.")
    print(f"{repr}")

    model = AutoModelForMaskedLM.from_pretrained(repr)
    tok = AutoTokenizer.from_pretrained(repr)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=True, mlm_probability=0.15
    )

    dataset = LineByLineTextDataset(
            tokenizer=tok,
            file_path=args.file,
            block_size=512
    )

    fraction = args.valid

    ind = list(range(len(dataset)))
    np.random.shuffle(ind)
    n_samples = int((1 - fraction) * len(dataset))
    train = deepcopy(dataset)
    test = deepcopy(dataset)
    train.examples = [dataset.examples[i] for i in ind[:n_samples]]
    test.examples = [dataset.examples[i] for i in ind[:n_samples]]

    training_args = TrainingArguments(
        output_dir=args.output,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_gpu_train_batch_size=args.batch_size,
        save_steps=10000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train,
        eval_dataset=test
    )
    tok.save_pretrained(str(pathlib.Path(args.output)) ) # So you can use the tokenizer with a checkpoint
    trainer.train()
    trainer.save_model(str(pathlib.Path(args.output)) )