import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

def read_data(
    for_english,
    for_both,
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
    if for_both:
        col_to_del_set = set(col_to_del) | set(["text_pl", "text"])
        labels = set(df.columns.tolist()) - col_to_del_set


        df = pd.concat([
            df[list(set(['text']) | labels)].rename(columns={'text': 'text'}),
            df[list(set(['text_pl']) | labels)].rename(columns={'text_pl': 'text'})
        ])
        # df.drop(columns=col_to_del, inplace=True)

        df.reset_index(drop=True, inplace=True)
        df = df[['text'] + list(labels)]
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df.replace({0: False, 1: True})
        labels = df.columns.tolist()[1:]

        size = int(df.shape[0] * proc_of_ds)
        random_rows = df.sample(n=size, replace=False)
        return random_rows, labels

    if for_english:
        col_to_del = ["text_pl"] + col_to_del
        df.drop(columns=col_to_del, inplace=True)
    else:
        col_to_del = ["text"] + col_to_del
        df.drop(columns=col_to_del, inplace=True)
        df.rename(columns={"text_pl": "text"}, inplace=True)

    df = df.replace({0: False, 1: True})
    labels = df.columns.tolist()[1:]

    size = int(df.shape[0] * proc_of_ds)
    random_rows = df.sample(n=size, replace=False)
    return random_rows, labels


def split_dataset(df):
    train_df, test_val_df = train_test_split(df, test_size=0.3, random_state=42)
    test_df, val_df = train_test_split(test_val_df, test_size=0.5, random_state=42)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    val_dataset = Dataset.from_pandas(val_df)

    del df, train_df, test_val_df, val_df

    return DatasetDict(
        {"train": train_dataset, "test": test_dataset, "validation": val_dataset}
    )


def make_tokenizer(labels, tokenizer_name):
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, return_token_type_ids="token_type_ids"
    )
    return tokenizer, id2label, label2id


def preprocess_data(examples, tokenizer, labels):
    text = examples["text"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]
    encoding["labels"] = labels_matrix.tolist()
    return encoding

