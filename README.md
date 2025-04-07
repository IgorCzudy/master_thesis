###### FACULTY OF COMPUTING AND TELECOMMUNICATION

### Emotion detection challenges in English to Polish parallel corpus

#### inż. Igor Czudy, 145198

###### POZNAŃ 2024


# Abstract

### English

The present study attempts to create a new Polish-language dataset used to predict a wide range of
emotions. Automatic translation tools were used to translate the already existing qualitative English
data set with is _GoEmotions: A Dataset of Fine-Grained Emotions_ [7]. This work seeks to determine
the quality of the newly created data and provides baseline models for the prediction of 27 emotions.
Furthermore, three distinct automatic translation tools were evaluated in this study. The tool that
demonstrated the most optimal performance, _DeepL_ , was subsequently employed.

### Polish

W niniejszej pracy podjęto próbę stworzenia nowego polskojęzycznego zbioru danych wykorzystywanego
do przewidywania szerokiego zakresu emocji. Narzędzia do automatycznego tłumaczenia zostały wyko-
rzystane do przetłumaczenia już istniejącego jakościowego zbioru danych w języku angielskim, którym
jest _GoEmotions: A Dataset of Fine-Grained Emotions_ [7]. Niniejsza praca ma na celu określenie jakości
nowo utworzonych danych i zapewnia bazowe modele do przewidywania 27 emocji. Ponadto w bada-
niu oceniono trzy różne narzędzia do automatycznego tłumaczenia. Następnie użyto narzędzia, które
wykazało najlepszą jakość tłumaczeń, którym jest _DeepL_.



## Contents

- 1 Introduction
- 2 Theoretical background
   - 2.1 Introduction
   - 2.2 Historical background
   - 2.3 State-of-the-art approach
   - 2.4 BERT
   - 2.5 Hyperparameters
   - 2.6 Loss function
   - 2.7 Metrics
      - 2.7.1 Accuracy
      - 2.7.2 Confusion Matrix
      - 2.7.3 Precision, recall, f1-score
      - 2.7.4 Dataset split
- 3 Dataset
   - 3.1 Translation
      - 3.1.1 DeepL translation
   - 3.2 Data analysis
      - 3.2.1 Emotion distibution
      - 3.2.2 Emotion correlation
      - 3.2.3 Comments text analysis
      - 3.2.4 Repetitions
      - 3.2.5 Keywords extraction
         - KeyBERT
         - TF-IDF
- 4 Experiments
   - 4.1 HerBERT model
      - 4.1.1 Prediction examples
   - 4.2 mDeBERTaV3 model
      - 4.2.1 DeBERTaV3 models results
         - DeBERTaV3 models results for individual classes
         - DeBERTaV3 models learning process
- 5 Conclusion
- Bibliography


```
The entire work can be found in the file `master_thesis.pdf`
```