# Sentiment Analysis Using RNN

This project implements a sentiment analysis model using a Recurrent Neural Network (RNN) to classify movie reviews as positive or negative.

## Project Overview

This model is trained on a dataset of movie reviews and aims to predict the sentiment of a given review. The dataset consists of text reviews, which are pre-processed and used to train the model. The model leverages an RNN architecture to learn patterns in the sequence of words in the review text.

### Key Features:
- **RNN Architecture**: The model uses a simple RNN to process the input text sequences.
- **Sentiment Classification**: The model classifies reviews into two categories: Positive and Negative.
- **Performance Metrics**: The model achieves high accuracy with precision, recall, and F1-score all around 0.86.

## Dataset

The dataset used for training and testing consists of labeled movie reviews, where each review is marked as either `Positive` or `Negative`.

- **Training Set**: 80% of the dataset.
- **Test Set**: 20% of the dataset.

## Dependencies

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn

[[4150  811]
 [ 625 4414]]


              precision    recall  f1-score   support

    Negative       0.87      0.84      0.85      4961
    Positive       0.84      0.88      0.86      5039

    accuracy                           0.86     10000
   macro avg       0.86      0.86      0.86     10000
weighted avg       0.86      0.86      0.86     10000


## Sample Predictions

Here are some sample predictions made by the Sentiment Analysis model:

1. **Review:** The movie was a waste of time, very boring and dull.  
   **Predicted Sentiment:** Negative  
   **Confidence:** 99.21%

2. **Review:** Absolutely loved the plot and acting, what a masterpiece!  
   **Predicted Sentiment:** Positive  
   **Confidence:** 69.64%

3. **Review:** The movie was decent but had its flaws.  
   **Predicted Sentiment:** Negative  
   **Confidence:** 76.97%

4. **Review:** Terrible direction but the actors tried their best.  
   **Predicted Sentiment:** Negative  
   **Confidence:** 73.80%

5. **Review:** It was neither good nor bad, just average.  
   **Predicted Sentiment:** Negative  
   **Confidence:** 97.48%
