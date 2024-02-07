Dominant Lead Type Classification Application of Undersampling Technique

This code modifies the dataset for dominant leaf type classification using undersampling.
Undersampling is a solution for imbalanced data that reduces data points of the dominant class while keeping the minority class unchanged. 

The code takes takes in a folder of images and labels. It iterates through all labels to select the ones with less than X% of Y-class pixels. In this context, X is a percentage value (defaul = 70%). Y is a selected class (default = Non-Tree). For the label images that validate the threshhold, the corresponding images are selected and added to a new dataset folder. This new dataset is considered when training the model. 

This code was created by Jemma Johnson and modified by Dr. Qian Song for the Remote Sensing Seminar Project. 
