# Big-Pixel

# Dataset
Kaggle Draper Satellite Chronology Dataset. In total there are 344 locations. Each location has 5 images. Total: 1720 images.
For each location 3 images are used for Training and 2 images are used for Validation. Training: 1032 images. Validation: 688 images.

https://www.kaggle.com/c/draper-satellite-image-chronology

# Feature Extraction 
Look into Feature Extraction Folder first too see how to feature extract using Keras/Caffe.


# Finetuneing Folder:
Shows how to Finetune in Keras. 

ResNet50finetune.py contains the abstracted network that needs to be finetuned. You will add the top layer.

myresnet50.py is exactly the same as ResNet50 from Keras but the mode for BatchNormalization is changed to 1.

How to Finetune in Caffe:
http://cs231n.stanford.edu/slides/2016/winter1516_lecture12.pdf 
Starting from slide 11

# Experiments:
https://docs.google.com/a/eng.ucsd.edu/document/d/1xGTAIGoygsyUx7va-xoh0cKTAov1ASoxVmfiBuM35UA/edit?usp=sharing

# Feature Subselection and Concatenation Folder:
   1) Contains code to use Random Forest for feature subselection
   2) Drawing from uniform distribution for feature subselection
   3) Concenating Feature sets together
    
# Extra Folder:
    How to plot the Validation Accuracy/Training loss using log.txt file. Used for Caffe.

# Test:
	Contains SVC classifier for UC Merced Test set.

    
 
