#%% imports
import numpy as np
import pandas as pd
import os.path

XTrainCSVFilename = 'XTrainSignale.csv'
YTrainCSVFilename = 'YTrainSignale.csv'
XTestCSVFilename = 'XTestSignale.csv'
TrainDataZipFilename = 'TrainDataSignale.zip'
TestDataZipFilename = 'XTestSignale.zip'

url_prefix = 'https://hs-bochum.sciebo.de/s/2DS5EFSrhJqUTvA/download?path=%2FUebungen%2Fdata&files='

#%% load data
class DownloadError(Exception):
    def __init__(self, msg, url=None, filename=None):
        super().__init__(msg)
        self.url = url
        self.filename = filename

def download_and_unzip(filename):
    # check if filename exists, otherwise download it first
    if not os.path.isfile(filename):
        from urllib.request import urlretrieve
        url = url_prefix + filename
        print(f'Downloading {filename} from {url}.')
        try:
            urlretrieve(url, filename)
        except Exception as e:
            raise DownloadError('Could not Download the file', url, filename) from e

    print(f'Extracting CSV file(s) from {filename}.')
    from zipfile import ZipFile
    with ZipFile(filename, 'r') as zip_file:
        zip_file.extractall()

def load_data():
    # Check if CSV files exist, download and unzip otherwise
    try:
        if not os.path.isfile(XTrainCSVFilename) or not os.path.isfile(YTrainCSVFilename):
            download_and_unzip(TrainDataZipFilename)
    except DownloadError as e:
        raise DownloadError(f'Could not find CSV file {XTrainCSVFilename} or {YTrainCSVFilename} and downloading {e.filename} from the following URL failed. Please provide the data.\n{e.url}') from e.__cause__

    try:
        if not os.path.isfile(XTestCSVFilename):
            download_and_unzip(TestDataZipFilename)
    except DownloadError as e:
        raise DownloadError(f'Could not find CSV file {XTestCSVFilename} and downloading {e.filename} from the following URL failed. Please provide the data.\n{e.url}') from e.__cause__

    print('Reading CSV files')
    XTrain = np.loadtxt('XTrainSignale.csv')
    XTrain = XTrain.reshape(-1, 3, XTrain.shape[-1]).transpose(0, 2, 1)

    XTest = np.loadtxt('XTestSignale.csv')
    XTest = XTest.reshape(-1, 3, XTest.shape[-1]).transpose(0, 2, 1)

    YTrainInt = pd.read_csv('YTrainSignale.csv', dtype=int, header=None, squeeze=True)

    return XTrain, YTrainInt, XTest

XTrain, YTrainInt, XTest = load_data()
XTrain = np.loadtxt('XTrainSignale.csv')
XTrain = XTrain.reshape(-1, 3, XTrain.shape[-1]).transpose(0, 2, 1)
YTrainInt = pd.read_csv('YTrainSignale.csv', dtype=int, header=None, squeeze=True)
XTest = np.loadtxt('XTestSignale.csv')
XTest = XTest.reshape(-1, 3, XTest.shape[-1]).transpose(0, 2, 1)
YTrainOneHot = pd.get_dummies(YTrainInt)
noOfClasses=4
#%% analysis
print('\nClass counts:', YTrainInt.value_counts(), sep='\n')
print('==> no balancing required.\n')

# Hier ihr Beitrag / CNN