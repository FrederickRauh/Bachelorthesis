from utils import dataframeManager as dam, directoryManager as dm

def load_result(file_path):
    result_json = dam.load_datafram_from_path(file_path)
    return result_json.confusion_mat[0]

