# sentiment-analysis-of-recipe-reviews

In this project I used a dataset from Kaggle consisting of over 1 million reviews for recipes scraped from GeniuKitchen. I have taken the text data and split it from the original dataset. In the SA_Processing notebook I have read in the reviews and sampled them to create a balanced dataset. I have visualised the different classes before cleaning the text data with RegEx. <br> Initally, I only solved for a binary problem (positive or negative reviews) but thought it would be interesting to look at positive, neutral and negative reviews and solve as a multi-class problem. Count Vectorizer from NLTK was used and several different classifiers were used to predict the classes. Stemming and TF-IDF was employed to try and improve model ROC AUC scores. The final model scores an overall ROC score of 89%, and the individual classification scores from the Logistic Regression using One-vs-Rest method was:
* Negative: 90%
* Neutral: 81%
* Positive: 95%

## Notes
<br>My original blog for the initial part of this project can be found here: https://medium.com/analytics-vidhya/analysing-recipe-reviews-with-python-for-natural-language-processing-cf35cf3acda0

Amongst many resources it would not be possible without the use of these blogs:
* https://medium.com/factory-mind/regex-tutorial-a-simple-cheatsheet-by-examples-649dc1c3f285
* https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184
* https://towardsdatascience.com/sentiment-analysis-with-python-part-2-4f71e7bde59a
* https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
* https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f

