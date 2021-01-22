# Deep-Learning-Raman-Spectroscopy

## Goal of the project
The purpose of this repository is to use Transfer Learning in order to classify patients affected by Amyotrophic Lateral Sclerosis using Raman Spectroscopy.

![Project Presentation](https://github.com/nsgln/Deep-Learning-Raman-Spectroscopy/blob/main/ProjectPresentation.png)


## Dataset :
The main dataset used in this project is composed of 393 spectra belonging to 20 patients.
affected by Amyotrophic Lateral Sclerosis (ALS) and 198 to 10 healthy ones (CTRL).
The data can be found [here](https://github.com/nsgln/Deep-Learning-Raman-Spectroscopy/tree/main/Raman_Data).

Notice that the pretrained model for the Transfer Learning experiments come from a bacteria dataset that you can found [here](https://github.com/csho33/bacteria-ID/blob/master/data/data.md).

## The project structure :
It is composed of 3 directories:
1) **Bacteria_TL** - all the files comes from [here](https://github.com/csho33/bacteria-ID) and is composed of : 
    - 3 pretrained models i.e saved parameters for pre-trained CNN (*pretrained_model.ckpt*, *finetuned_model.ckpt* and *clinical_pretrained_model.ckpt* )
    - datasets.py - contains code for setting up datasets and dataloaders for spectral data
    - resnet.py - contains ResNet CNN model class
    - training.py - contains code for training CNN and making predictions
2) **Project Report** - contains the final report of the project written in [LaTeX](https://www.latex-project.org/) using  MiKTeX and editing with [Texmaker](https://www.xm1math.net/texmaker/)
3) **Raman_Data** - 2 sub-folders containing the spectra of [ALS](https://github.com/nsgln/Deep-Learning-Raman-Spectroscopy/tree/main/Raman_Data/ALS) et [CTRL](https://github.com/nsgln/Deep-Learning-Raman-Spectroscopy/tree/main/Raman_Data/CTRL) + a CSV file summing up the patient IDs and the samples IDs
4) **checkpoints** - it contains the checkpoints of the features extractor models developped in the notebook 4 and 6 

It is also composed of 5 jupyter notebooks :
 - **1_test_models_dataset.ipynb** - file to *load the data*, *pre-process it* (removing negative values and features selection), *plot some of spectra* and *predict* on simple (LogisticRegression, DecisionTree) and a bit more complex (SVM, RandomForest) Machine Learning (ML) models using different splitting techniques (LeaveOneGroupOut and GroupKFold)
 - **2_predictions_with_pretrained_models.ipynb** - Using the pretrained models of Bacteria-ID on our dataset to make some predictions using average accuracy and standard deviation.
 - **3_fine_tuning_experiments.ipynb** - After a custom splitting dataset technique producing a "finetunable" set and a "test" set, we finetune the predicted models in order to determine the best model and increase our average accuracy.
 - **4_features_extraction.ipynb** - Using each pretrained models of Bacteria-ID, features are extracted ("representing" our data), from different layer. Then, two different models are tested on these features : a classical model and a deep one. 
 - **5_data_augmentation.ipynb** - Using some data augmentation technique on spectral data (offset, multiplication and Gaussian noise), the finetuned models seems to obtained better results.
 - **6_data_augmentation_with_features_extractor** - The methods of data augmentation previously used on the finetuned models are now applied to the features extraction method.


Finally the remaining files are :
 - *data_loader.py* - a python file to load the data (factorization of code
 - *extractor.py* - a modification of the resnet.py file class to extract features (by removing the final layers)
    
    
## Requirements

The code in this repo has been testing with python 3.6.9 and python 3.6.12 using Anaconda Python distribution.

## Reference papers
- [Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning](https://www.nature.com/articles/s41467-019-12898-9)
- [Deep convolutional neural networks for Raman spectrum recognition : a unified solution](https://pubs.rsc.org/en/content/articlelanding/2017/an/c7an01371j#!divAbstract)
- [Data Augmentation of Spectral Data for CNN](https://arxiv.org/pdf/1710.01927.pdf)

