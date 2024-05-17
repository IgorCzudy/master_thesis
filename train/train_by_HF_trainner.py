from .tokenization import read_data, split_dataset, make_tokenizer, preprocess_data
import torch
import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)
from transformers import (
    BitsAndBytesConfig,
    EvalPrediction,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
import mlflow
import os

def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    f1_scores_per_class = f1_score(
        y_true=y_true, y_pred=y_pred, average=None
    )
    f1_per_class_dict = {
        f"f1_class_{id2label[i]}_validation": score
        for i, score in enumerate(f1_scores_per_class)
    }

    # confusion_mat = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    metrics = {"f1_micro": f1_micro_average,
               "f1_macro": f1_macro_average,
               "roc_auc": roc_auc,
               "accuracy": accuracy,
               **f1_per_class_dict, }
              #  "confusion_matrix": np.array2string(confusion_mat)}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


def train_easy(
    model_name,
    encoded_dataset,
    tokenizer,
    labels,
    id2label,
    label2id,
    for_english,
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

    metric_name = "f1_micro"
    args = TrainingArguments(
        f"{model_name}-finetuned-number-of-epochs-{num_train_epochs}-batch-size-{batch_size}-lr-{learning_rate}-wd-{weight_decay}-for_english-{for_english}",
        # evaluation_strategy="steps",
        # save_strategy="steps",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        run_name=f"{model_name}-finetuned-number-of-epochs-{num_train_epochs}-batch-size-{batch_size}-lr-{learning_rate}-wd-{weight_decay}",
        report_to=["mlflow"],
        # eval_steps = 100
    )
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


def train_lora(
    model_name,
    encoded_dataset,
    tokenizer,
    labels,
    id2label,
    label2id,
    lora_config,
    quantization_config,
    for_english,
    batch_size=16,
    learning_rate=5e-5,
    num_train_epochs=5,
    weight_decay=0.01,
):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        problem_type="multi_label_classification",
        quantization_config=quantization_config,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.print_trainable_parameters()

    metric_name = "f1_micro"
    args = TrainingArguments(
        f"{model_name}-finetuned-number-of-epochs-{num_train_epochs}-batch-size-{batch_size}-lr-{learning_rate}-wd-{weight_decay}-for_english-{for_english}-lora",
        # evaluation_strategy="steps",
        # save_strategy="steps",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        run_name=f"{model_name}-finetuned-number-of-epochs-{num_train_epochs}-batch-size-{batch_size}-lr-{learning_rate}-wd-{weight_decay}",
        report_to=["mlflow"],
        # eval_steps = 100
    )
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
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    predicted_labels = [
        id2label[idx] for idx, label in enumerate(predictions) if label == 1.0
    ]
    return predicted_labels

# if __name__ == "__main__":
#     FOR_ENGLISH = False
#     PROC_OF_DS = 0.5
#     TOKENIZER_NAME = "allegro/herbert-base-cased" #"bert-base-uncased"
#     BATCH_SIZE = 64
#     LR = 5e-3
#     NUM_EPOCH = 5
#     WEIGHT_DECAY=0.01

#     df, labels = read_data(for_english=FOR_ENGLISH, proc_of_ds=PROC_OF_DS)
#     dataset = split_dataset(df)
#     print(dataset.shape)

#     tokenizer, id2label, label2id = make_tokenizer(labels, tokenizer_name=TOKENIZER_NAME)

#     encoded_dataset = dataset.map(
#         lambda examples: preprocess_data(examples, tokenizer, labels),
#         batched=True,
#         remove_columns=dataset["train"].column_names,
#     )

#     encoded_dataset.set_format("torch")

#     os.environ['MLFLOW_TRACKING_USERNAME'] = "IgorCzudy"
#     os.environ['MLFLOW_TRACKING_PASSWORD'] = "42876d647d0b11f68a0aae1c1c41bf76214e41db"
#     mlflow.set_tracking_uri('https://dagshub.com/IgorCzudy/master_thesis.mlflow')
#     mlflow.set_experiment("TEST")

#     trainer = train_easy(TOKENIZER_NAME,encoded_dataset,tokenizer,labels,id2label,label2id,batch_size=BATCH_SIZE,learning_rate=LR, num_train_epochs=NUM_EPOCH,weight_decay=WEIGHT_DECAY,)
#     mlflow.pytorch.log_model(trainer.model, f"saved_model_{TOKENIZER_NAME}")

