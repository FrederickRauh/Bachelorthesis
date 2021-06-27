import numpy as np

# util part
def get_correct_array_form(array):
    x = np.array(array)
    nsamples, nx, ny = x.shape
    return x.reshape((nsamples, nx * ny))


# def get_correct_feature_array(files):
#     x = []
#     for file in files:
#         file_path = file
#         wav_path = file_path.replace('.csv', '.wav2')
#         features = extract_mfcc_from_file(wav_path)
#         # decide which feature array to use
#         features_small = features[1: 3, :]
#         feature_array = features_small
#         x.append(feature_array)
#     return x
    # return get_correct_array_form(x)