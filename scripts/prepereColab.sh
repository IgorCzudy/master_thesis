pip install torch==2.2.1 sacremoses mlflow dvc datasets accelerate -U transformers[torch] --quiet
dvc remote add --default master_thesis /content/drive/MyDrive/master_thesis
dvc pull
