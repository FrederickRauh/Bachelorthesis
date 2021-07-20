# Results of 3 Datasets:
- local contains 4 Speakers
- ba contains 20 Speakers out of the Voxceleb1 dataset
- bigba contains 100 Speakers out of the Voxceleb1 dataset

# Structure
Filename contains library that was used for feature extraction and how many mfcc were used.
Multiple versions are being collected to obtain a mean for each mfcc count and the resulting confusion_mat.

# First impression
- local
    - SVM: Every training and predicition case for the same amount of mfcc is absolutly equal (best mfcc = 20). 
    - GMM: Not the same: 14, 15,16, 18, 19, 20
    - GMM: noteworthy 17 was in both versions 100% accurate 