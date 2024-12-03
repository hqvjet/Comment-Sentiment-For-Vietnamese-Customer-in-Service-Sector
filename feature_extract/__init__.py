import pandas as pd
import numpy as np

from feature_extract.normalize import useNormalize
from feature_extract.lemma import useLemma
from feature_extract.tokenize import useTokenize
from feature_extract.remove_stopword import removeStopword
from feature_extract.identify import useIdentify
from feature_extract.extract_feature import extractFeature

def getDataset(file_path):
    try:
        # return pd.read_excel(file_path, engine='openpyxl')
        return pd.read_csv(file_path)
    except Exception as error:
        print('ERROR WHILE READING DATASET')
        print(error)

def makeData(data):
    titles = data['title'].apply(str)
    contents = data['text'].apply(str)

    return titles, contents
        
def useFeatureExtractor(device):
    data = getDataset('res/true_data.csv')
    title, content = makeData(data)

    title = useNormalize(title)
    content = useNormalize(content)

    title = useLemma(title)
    content = useLemma(content)

    title = useTokenize(title)
    content = useTokenize(content)

    title = removeStopword(title)
    content = removeStopword(content)
 
    key = input('Choose feature extractor method:\n1. PhoBERT\n2. PhoW2V\nYour Input: ')
    
    if key == '1':
        model = 'phobert'
    elif key == '2':
        model = 'phow2v'
    else:
        print('Wrong method, please try again')

    if key == '1':
        title, title_attention = useIdentify(title)
        content, content_attention = useIdentify(content)
        title = extractFeature(device, title, title_attention, model=model)
        content = extractFeature(device, content, content_attention, model=model)
    else:
        title = extractFeature(device, title, model=model)
        content = extractFeature(device, content, model=model)

    np.save(f'res/features/{model}_title_features.npy', title.cpu())
    np.save(f'res/features/{model}_content_features.npy', content.cpu())
