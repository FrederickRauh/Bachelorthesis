import inspect

from utils.config import DEBUG


def log(message):
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    filename = module.__file__
    feature_extraction = 'frontend'
    train = '_model'
    predict = '_predictor'

    if DEBUG.LOGLEVEL == 0:
        pass
    if DEBUG.LOGLEVEL == 1:
        if filename.__contains__('main'):
            log_message(message)
    if DEBUG.LOGLEVEL == 2:
        if filename.__contains__('main') or filename.__contains__(predict):
            log_message(message)
    if DEBUG.LOGLEVEL == 3:
        if filename.__contains__('main') or filename.__contains__(predict) or filename.__contains__(train):
            log_message(message)
    if DEBUG.LOGLEVEL == 4:
        if filename.__contains__('main') or filename.__contains__(predict) or filename.__contains__(
                train) or filename.__contains__(feature_extraction):
            log_message(message)
    if DEBUG.LOGLEVEL > 4:
        log_message(message)


def log_message(message):
    for string in message:
        string = str(string)
        if string.__contains__(':') or string.__contains__(','):
            print(string, end=' ')
        else:
            print(string, end='')
    print()
