# Final Project - Krish Rai and Taimoor Qureshi: Predicting Positive vs Negative Emotions for Facial Imaging Data

Dataset: https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition
How to use Kaggle dataset in Google Colab:
https://www.geeksforgeeks.org/how-to-import-kaggle-datasets-directly-into-google-colab/
download opendatasets to download kaggle dataset, pandas as well
On kaggle account - go to settings, scroll down click new API token, and download kaggle.json - open it up and store username and password
%pip install opendatasets pandas
import opendatasets as od
import pandas
 
 the following dataset is used - download if not working to see, but when you run the code provide the username and password from kaggle.json earlier in the given box that pops up in output
od.download(
    "https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition")


## Problem: To identify positive or negative emotions through facial imaging data (where various emotions including Negative: sadness, anger, fear, disgust and Positive: happy, surprise, neutral. 
## Question: Can we predict a binary positive or negative from a person’s imaging data?
## Input features: gray scaled images of various faces with vastly different emotions (in jpeg format)

Our project aimed to leverage neural network capabilities for predicting image data, with a particular interest in interpreting facial expressions to discern whether an individual expresses positivity or negativity in an image. We sought to address the challenge of identifying positive or negative emotions from facial imaging data, encompassing emotions such as sadness, anger, fear, disgust (negative), and happiness, surprise, neutral (positive). Our input features comprised grayscale images of diverse faces in JPEG format.

Despite challenges in sourcing suitable facial emotion data, we discovered a viable dataset on Kaggle, containing JPEG files categorized by specific emotions. Initially, we encountered limitations with a dataset of only 18 participants, where each demonstrated varied facial expressions. This lack of diversity resulted in poor accuracy during testing. However, our subsequent acquisition of a dataset comprising 28,000 rows significantly improved our performance and prediction accuracy: Face expression recognition dataset (kaggle.com)

To process the facial images consistently and efficiently, we resized them to a 28 x 28 array, maintaining uniformity.

Our primary objective was to classify each facial emotion into either a positive or negative category. We chose to implement a Neural Network using TensorFlow (sk.learn was used as well) due to its suitability for image classification tasks. Initially, our Neural Network proved overly complex for the training data, leading to overfitting and unrealistically high accuracies. Subsequent adjustments, including shuffling our training data (emotion label column out of order) and fine-tuning the model, yielded more reasonable accuracy metrics. Ultimately, our model achieved a test accuracy of approximately 0.59-0.60.

For model evaluation, we employed both a confusion matrix and ROC curve analysis. Our model exhibited a higher true negative rate and a lower false positive rate, indicating its effectiveness in discerning negative emotions. Additionally, the ROC area under the curve (AUC) measured at 0.66, reflecting the overall performance of our model in distinguishing between positive and negative emotions.

We also wanted to take a few random images from the test set and see if our model was able to distinguish between positive and negative emotions. If the accuracy of an image in the predicted test set was greater than 0.6, we would classify the facial emotion as positive, else, negative. We were able to see an accurate distinction more than half the time in the test dataset. 

Limitations: Our Accuracy for training and testing is still under 0.65, which is far from perfect. This likely needs to be improved. There were still some images that were not fully clear which emotion they were representing (although much better than the initial dataset we used), but the goal is to get this to be better. This would likely require going through the data and seeing which images made it cause errors. Affectnet for example is a large database (0.4 million?) that has facial emotion data. This may be better for this project. We tried adding more layers and tuning other parameters, but it did not make much of a difference and we did not want to overfit again after out initial struggles. We also wanted to stick to the neural network model compared to other binary classification models like logistic regression. However, the accuracy did improve and the ROC curve showed that it was much much better than before (our initial dataset was 0.5 and was basically a random classifier).

Citations: We used mainly Google Colab’s AI assistant, GitHub Co-Pilot, and ChatGPT for coding and debugging assistance. Data is from Face expression recognition dataset (kaggle.com).
