from tokenization import read_data, split_dataset, make_tokenizer, preprocess_data
from CustomTransformer import CustomTransformer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelF1Score, MultilabelAccuracy, MultilabelPrecision, MultilabelRecall
from torchmetrics.wrappers import ClasswiseWrapper
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
)
from transformers import AdamW
import mlflow
from transformers.integrations import MLflowCallback
import os
from tqdm.auto import tqdm


def test_slow(
    custom_transformer,
    eval_dataloader,
    id2label,
    criterion,
    threshold=0.5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    custom_transformer.eval()
    with torch.no_grad():
        total_loss = 0.0
        predictions_list = []
        labels_list = []

        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = custom_transformer(
                batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            loss = criterion(outputs, batch["labels"])
            total_loss += loss.item()

            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(outputs)  # .squeeze().to(device))
            predictions = torch.zeros_like(probs).to(device)
            predictions[probs >= threshold] = 1

            predictions_list.extend(predictions.cpu().numpy())
            labels_list.extend(batch["labels"].cpu().numpy())
            # predictions_list.append(predictions.cpu().numpy())
            # labels_list.append(batch["labels"].cpu().numpy())
    custom_transformer.train()
    average_loss = total_loss / len(eval_dataloader)

    # labels_list = [0,0,0,0,1 ... ] predictions_list=[0,0,1,0,1..]

    f1_micro_average = f1_score(
        y_true=labels_list, y_pred=predictions_list, average="micro"
    )
    f1_macro_average = f1_score(
        y_true=labels_list, y_pred=predictions_list, average="macro"
    )
    accuracy = accuracy_score(y_true=labels_list, y_pred=predictions_list)

    f1_scores_per_class = f1_score(
        y_true=labels_list, y_pred=predictions_list, average=None
    )
    f1_per_class_dict = {
        f"f1_class_{id2label[i]}_validation": score
        for i, score in enumerate(f1_scores_per_class)
    }
    roc_auc = roc_auc_score(
        y_true=labels_list, y_score=predictions_list, average="micro"
    )

    return {
        "f1_micro_validation": f1_micro_average,
        "f1_macro_validation": f1_macro_average,
        **f1_per_class_dict,  # Include F1 scores per class
        "roc_auc_validation": roc_auc,
        "accuracy_validation": accuracy,
        "loss_validation": average_loss,
    }


def count_f1micro(TP, FP, FN):
    if (TP + (1 / 2) * (FP + FN)) == 0:
        return  0
    return TP / (TP + (1 / 2) * (FP + FN))


def test_fast(custom_transformer, eval_dataloader, criterion, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        custom_transformer.eval()
        running_loss = 0.0
        TP = 0
        FP = 0
        FN = 0
        for i, batch in enumerate(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = custom_transformer(
                batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            loss = criterion(outputs, batch["labels"])
            running_loss += loss.item()

            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(outputs)  # .squeeze().to(device))
            predictions = torch.zeros_like(probs).to(device)
            predictions[probs >= threshold] = 1

            TP += torch.logical_and(predictions, batch["labels"]).sum().item()
            orr = torch.logical_xor(predictions, batch["labels"])
            FP += torch.logical_and(orr, predictions).sum().item()
            FN += torch.logical_and(orr, batch["labels"]).sum().item()

        average_loss = running_loss / len(eval_dataloader)

        f1_validation = count_f1micro(TP, FP, FN)

    return {
        "loss_validation": average_loss,
        "f1_micro_validation": f1_validation,
        "TP_validation": TP,
        "FP_validation": FP,
        "FN_validation": FN,
        "len_of_eval_dataloader": len(eval_dataloader),
    }

def test_by_torch_lightning(custom_transformer, eval_dataloader, criterion, num_classes, threshold=0.5):
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      custom_transformer.eval()
      with torch.no_grad():
          running_loss = 0.0

          metric_collection = MetricCollection({
                "precision_macro": MultilabelPrecision(num_labels=num_classes, average="macro"),
                "recall_macro": MultilabelRecall(num_labels=num_classes, average="macro"),
                "accuracy_micro": MultilabelAccuracy(num_labels=num_classes, average="micro"),
                "f1_score_none": ClasswiseWrapper(MultilabelF1Score(num_labels=num_classes, average=None), labels=labels),
                "f1_score_macro": MultilabelF1Score(num_labels=num_classes, average="macro"),
                "f1_score_micro": MultilabelF1Score(num_labels=num_classes, average="micro"),
          }).to(device)


          for i, batch in enumerate(eval_dataloader):
              batch = {k: v.to(device) for k, v in batch.items()}
              outputs = custom_transformer(
                  batch["input_ids"], attention_mask=batch["attention_mask"]
              )
              loss = criterion(outputs, batch["labels"])
              running_loss += loss.item()

              sigmoid = torch.nn.Sigmoid()
              probs = sigmoid(outputs)  # .squeeze().to(device))
              # predictions = torch.zeros_like(probs).to(device)
              # predictions[probs >= threshold] = 1
              # print(probs)
              # print(batch["labels"])

              metric_collection.update(probs, batch["labels"])
              # break

          metric_collection = metric_collection.clone(postfix='_validation')
          metric_collection_computed = metric_collection.compute()
      custom_transformer.train()

      metric_collection_computed_float = {key: value.item() if isinstance(value, torch.Tensor) else value for key, value in metric_collection_computed.items()}

      average_loss = running_loss / len(eval_dataloader)
      print(metric_collection_computed_float)
      return {
          "loss_validation": average_loss,
          **metric_collection_computed_float,
          }

def train_torch(
    train_dataloader,
    eval_dataloader,
    id2label,
    num_labels,
    checkpoint,
    lr,
    num_epoch,
    number_of_test,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    custom_transformer = CustomTransformer(
        checkpoint=checkpoint, num_labels=num_labels
    ).to(device)

    optimizer = AdamW(custom_transformer.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    num_training_steps = num_epoch * len(train_dataloader)
    params = {"loss_function": "BCEWithLogitsLoss",
              "optimizer": "AdamW",
              "total_numer_of_training_steps": num_training_steps}
    mlflow.log_params(params)

    progress_bar_train = tqdm(range(num_training_steps))
    step = 0
    f1_micro_test = MultilabelF1Score(num_labels=num_labels, average="micro").to(device)

    for epoch in range(num_epoch):
        running_loss = 0.0
        # custom_transformer.train()
        for i, batch in enumerate(train_dataloader):
            step+=1
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = custom_transformer(
                batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            loss = criterion(outputs, batch["labels"])

            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(outputs)
            predictions = torch.zeros_like(probs).to(device)
            predictions[probs >= 0.5] = 1

            f1_micro_test.update(predictions, batch["labels"])

            if (i+1)%100==0:#step % (num_training_steps // number_of_test) == 0:
                last_loss_train = running_loss / i
                mlflow.log_metrics({"loss_train": last_loss_train,
                                    "f1_micro_train": f1_micro_test.compute().item()}, step=step)

                print(
                    f"Epoch {epoch+1}/{num_epoch}, Step {step}/{num_training_steps}, last_loss_train: {last_loss_train}"
                )

                metrics = test_by_torch_lightning(custom_transformer, eval_dataloader, criterion, num_labels, threshold=0.5)
                mlflow.log_metrics(metrics, step=step)

                m = test_slow(custom_transformer, eval_dataloader,id2label,criterion,threshold=0.5,)
                print(m)
            progress_bar_train.update(1)

        print(
            f"Epoch {epoch+1}/{num_epoch}, Step {step}/{num_training_steps}, Loss: {running_loss/len(train_dataloader)}"
        )

        metrics = test_by_torch_lightning(custom_transformer, eval_dataloader, criterion, num_labels, threshold=0.5)
        mlflow.log_metrics(metrics, step=step)
        mlflow.pytorch.log_model(custom_transformer, f"custom_transformer_{checkpoint}_epoch:{epoch+1}")

if __name__=="__main__":
    FOR_ENGLISH = False
    PROC_OF_DS = 0.5
    TOKENIZER_NAME = "allegro/herbert-base-cased" #"bert-base-uncased"
    BATCH_SIZE = 64
    LR = 5e-3
    NUM_EPOCH = 5
    WEIGHT_DECAY=0.01

    df, labels = read_data(for_english=FOR_ENGLISH, proc_of_ds=PROC_OF_DS)
    dataset = split_dataset(df)
    print(dataset.shape)

    tokenizer, id2label, label2id = make_tokenizer(labels, tokenizer_name=TOKENIZER_NAME)

    encoded_dataset = dataset.map(
        lambda examples: preprocess_data(examples, tokenizer, labels),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    encoded_dataset.set_format("torch")

    os.environ['MLFLOW_TRACKING_USERNAME'] = "IgorCzudy"
    os.environ['MLFLOW_TRACKING_PASSWORD'] = "42876d647d0b11f68a0aae1c1c41bf76214e41db"
    mlflow.set_tracking_uri('https://dagshub.com/IgorCzudy/master_thesis.mlflow')
    mlflow.set_experiment("TEST")

    train_dataloader = DataLoader(
        encoded_dataset['train'], shuffle = True, batch_size = BATCH_SIZE, #collate_fn = data_collator
    )
    eval_dataloader = DataLoader(
        encoded_dataset['validation'], shuffle = True, batch_size = BATCH_SIZE,#collate_fn = data_collator
    )

    with mlflow.start_run() as run:
        mlflow.set_tag("mlflow.runName",f"Pretrained:{TOKENIZER_NAME}_for_english={FOR_ENGLISH}_pr_od_ds:{PROC_OF_DS}_batch_size:{BATCH_SIZE}")
        params = {
            "tokenizer_name": TOKENIZER_NAME,
            "procent_of_dataset": PROC_OF_DS,
            "epochs": NUM_EPOCH,
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "for_english": FOR_ENGLISH,
            "id2label": id2label,
            "data_set_size": dataset.shape
        }
        mlflow.log_params(params)

        train_torch(train_dataloader,
                    eval_dataloader,
                    id2label = id2label,
                    num_labels=len(id2label),
                    checkpoint = TOKENIZER_NAME,
                    lr = LR,
                    num_epoch=NUM_EPOCH,
                    number_of_test = 10)