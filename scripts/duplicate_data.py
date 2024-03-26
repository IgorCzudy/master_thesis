from datasets import load_dataset
import pandas as pd 
import json 


if __name__ == "__main__":
    dataset = load_dataset("go_emotions", "raw")
    go_emotion = dataset["train"].to_pandas()


    final_translation = pd.read_csv("data/final_translation.csv")
    assert (final_translation["id"] == final_translation["id_pl"]).all()

    print(f"{go_emotion.shape=}, {final_translation.shape=}")

    translation_maping = final_translation.set_index('text')['text_pl'].to_dict()
    with open('data/translation_maping.json', 'w') as fp:
        json.dump(translation_maping, fp)

    go_emotion["text_pl"] = go_emotion["text"].apply(lambda x: translation_maping[x])
    
    col = ["text", "text_pl"]+go_emotion.columns.to_list()[1:-1]
    go_emotion = go_emotion.reindex(columns = col)
    go_emotion.to_csv("data/translated_go_emotion.csv")

