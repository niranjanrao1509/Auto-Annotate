# Auto-Annotate
API to for text, image, audio annotation.

## Models Used
**Text**
- Single class Classification
  > Naive Bayes
  > Logistic Regression
- Sentiment
  > TextBlob
  > VanderSentiment
- Multi Class
  > Spacy
  > Magpie



**Image**
- Classification
  > YOLO
  > ResNet 152
- Segmentation
  > YOLO
  > Facebook Detectron
- Landmark Detection
  > DLIB


**Audio**
- Classification
  > CNN encoder decoder
  > RNN encoder decoder

Output is JSON format file with annotations in utils/flask/static exports
