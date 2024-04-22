import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import mlflow

def read_data(
    for_english=True,
    path="/content/master_thesis/data/translated_go_emotion.csv",
    proc_of_ds=1,
):
    df = pd.read_csv(path)

    col_to_del = [
        "Unnamed: 0",
        "id",
        "author",
        "subreddit",
        "link_id",
        "parent_id",
        "created_utc",
        "rater_id",
        "example_very_unclear",
    ]
    if for_english:
        col_to_del = ["text_pl"] + col_to_del
    else:
        col_to_del = ["text"] + col_to_del
    df.drop(columns=col_to_del, inplace=True)
    df = df.replace({0: False, 1: True})
    labels = df.columns.tolist()[1:]

    size = int(df.shape[0] * proc_of_ds)
    return df.iloc[:size, :], labels


def split_dataset(df):
    train_df, test_val_df = train_test_split(df, test_size=0.3, random_state=42)
    test_df, val_df = train_test_split(test_val_df, test_size=0.5, random_state=42)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    val_dataset = Dataset.from_pandas(val_df)

    return DatasetDict(
        {"train": train_dataset, "test": test_dataset, "validation": val_dataset}
    )


def make_tokenizer(labels, tokenizer_name="bert-base-uncased", for_english=True):
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer, id2label, label2id


def preprocess_data(examples, tokenizer, labels, for_english=True):
    if for_english:
        text = examples["text"]
    else:
        text = examples["text_pl"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()

    return encoding


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1

    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {"f1": f1_micro_average, "roc_auc": roc_auc, "accuracy": accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


def train(
    model_name,
    encoded_dataset,
    tokenizer,
    labels,
    id2label,
    label2id,
    batch_size=16,
    learning_rate=5e-5,
    num_train_epochs=5,
    weight_decay=0.01,
):

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    metric_name = "f1"
    args = TrainingArguments(
        f"{model_name}-finetuned-number-of-epochs-{num_train_epochs}-batch-size-{batch_size}-lr-{learning_rate}-wd-{weight_decay}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        raport_to=["mlflow"],
        # warmup_steps
        # logging_dir: Ścieżka do katalogu, w którym będą przechowywane dzienniki treningu, w tym miary wydajności i metryki.
        # report_to: Lista nazw zasobów, do których chcesz wysłać raporty, takie jak "wandb", "tensorboard", "mlflow", itp
        # push_to_hub=True,
    )

    with mlflow.start_run():
        mlflow.log_params({
            "model_name": f"{model_name}-finetuned-number-of-epochs-{num_train_epochs}-batch-size-{batch_size}-lr-{learning_rate}-wd-{weight_decay}",
            "num_train_epochs": num_train_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        })

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return trainer


def predict(
    trainer,
    tokenizer,
    id2label,
    text="I'm happy I can finally train a model for multi-label classification",
):

    encoding = tokenizer(text, return_tensors="pt")
    encoding = {k: v.to(trainer.model.device) for k, v in encoding.items()}

    outputs = trainer.model(**encoding)
    logits = outputs.logits
    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    # turn predicted id's into actual label names
    predicted_labels = [
        id2label[idx] for idx, label in enumerate(predictions) if label == 1.0
    ]
    return predicted_labels


# if __name__ == "__main__":
#     df, labels = read_data(for_english=True, proc_of_ds=0.005)
#     dataset = split_dataset(df)
#     print(dataset.shape)

#     tokenizer, id2label, label2id = make_tokenizer(labels, for_english=True)

#     encoded_dataset = dataset.map(
#         lambda examples: preprocess_data(examples, tokenizer, labels, for_english=True),
#         batched=True,
#         remove_columns=dataset["train"].column_names,
#     )

#     encoded_dataset.set_format("torch")

#     trainer = train(
#         encoded_dataset,
#         tokenizer,
#         labels,
#         id2label,
#         label2id,
#         for_english=True,
#         batch_size=16,
#     )

#     trainer.evaluate()

#     predicted_labels = predict(trainer, tokenizer, id2label)
#     print(f"{predicted_labels=}")
