from datasets import load_dataset
import pandas as pd 
import json 


if __name__ == "__main__":
    dataset = load_dataset("go_emotions", "raw")
    go_emotion = dataset["train"].to_pandas()


    with open('data/translation_maping.json') as json_file:
        translation_maping = json.load(json_file)

    go_emotion["text_pl"] = go_emotion["text"].apply(lambda x: translation_maping[x])
    
    col = ["text", "text_pl"]+go_emotion.columns.to_list()[1:-1]
    go_emotion = go_emotion.reindex(columns = col)
    go_emotion.to_csv("data/translated_go_emotion.csv")

