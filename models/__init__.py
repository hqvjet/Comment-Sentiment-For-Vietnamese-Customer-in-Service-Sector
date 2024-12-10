import torch
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
import queue
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
from models.LSTM import LSTM
from models.BiLSTM import BiLSTM
from models.CNN import CNN2d
from models.XGBoost import XGBoost
from models.FullyConnected import FC
from models.LR import LR
from models.CNN_LSTM import CNNnLSTM
from models.CNN_BILSTM import CNNnBiLSTM
from models.Transformer import Transformer
from models.GRU import GRU
from models.BiGRU import BiGRU
from constant import DRIVE_PATH

with open('models/global_config.json', 'r') as file:
    config = json.load(file)

batch_size = config.get('batch_size')
num_epoch = config.get('num_epoch')

def separate_equally_dataset(title, content, rating, NEG, NEU, POS):
    n = title.shape[0]

    data = {
        'title': title,
        'content': content,
        'rating': rating
    }

    neg_filter = {
        'title': [title for title, rating in zip(data['title'], data['rating']) if rating == 0],
        'content': [content for content, rating in zip(data['content'], data['rating']) if rating == 0],
        'rating': [rating for rating in data['rating'] if rating == 0]
    }
    neu_filter = {
        'title': [title for title, rating in zip(data['title'], data['rating']) if rating == 1],
        'content': [content for content, rating in zip(data['content'], data['rating']) if rating == 1],
        'rating': [rating for rating in data['rating'] if rating == 1]
    }
    pos_filter = {
        'title': [title for title, rating in zip(data['title'], data['rating']) if rating == 2],
        'content': [content for content, rating in zip(data['content'], data['rating']) if rating == 2],
        'rating': [rating for rating in data['rating'] if rating == 2]
    }

    print(len(neg_filter['title']), len(neu_filter['title']), len(pos_filter['title']))
    
    temp_title, temp_content, temp_rating = [], [], []
    def append(index, dict):
        temp_title.append(dict['title'][index])
        temp_content.append(dict['content'][index])
        temp_rating.append(dict['rating'][index])

    i = 0
    while(NEG > i // 3 and NEU > i // 3 and POS > i // 3):
        if i % 3 == 0:
            append(i // 3, neg_filter)
        elif i % 3 == 1:
            append(i // 3, neu_filter)
        elif i % 3 == 2:
            append(i // 3, pos_filter)

        i += 1

    print(temp_rating)

    return temp_title, temp_content, temp_rating

def startTraining(device):
    key = input('Choose feature source:\n1. PhoBERT\n2. PhoW2V\nYour Input: ')

    print('Loading extracted feature')

    if key == '1':
        source = 'phobert'
    elif key == '2':
        source = 'phow2v'
    else:
        print('Wrong source, please try again')

    title = np.load(f'res/features/{source}_title_features.npy')
    content = np.load(f'res/features/{source}_content_features.npy')
    data = pd.read_csv('res/true_data.csv')
    rating = data['rating'].apply(int)
    rating -= 1

    o, t, tr = 0, 0, 0

    for r in rating:
        if r == 0:
            o += 1
        elif r == 1:
            t += 1
        else:
            tr += 1

    print(f'Negative: {o}, Neural: {t}, Positive: {tr}')

    # title, content, rating = separate_equally_dataset(title, content, rating, o, t, tr)

    o, t, tr = 0, 0, 0

    for r in rating:
        if r == 0:
            o += 1
        elif r == 1:
            t += 1
        else:
            tr += 1

    print(f'After separate: Negative: {o}, Neural: {t}, Positive: {tr}')

    title = torch.tensor(title)
    content = torch.tensor(content)
    rating = torch.tensor(rating)
    rating = torch.nn.functional.one_hot(rating, num_classes=3)

    print('Loading Done')
    print(f'Title Shape: {title.size()}')
    print(f'Content Shape: {content.size()}')
    print(f'Label Shape: {rating.size()}')

    key = input('Use Title and Content ?\n1. Yes\n2. No\nYour Input: ')
    if key == '1':
        data = torch.cat((title, content), dim=-1)
    else:
        data = content
    useTitle = False if key == '2' else True

    key = input('Choose one of these classification to train:\n1. LSTM\n2. BiLSTM\n3. XGBoost\n4. LG\n5. Ensemble CNN LSTM\n6. Ensemble CNN BiLSTM\n7. GRU\n8. BiGRU\n9. Transformer\n10.CNN\nYour Input: ')
    emb_tech = 1 if source == 'phobert' else 2
    input_shape = data.size()

    if key == '1':
        train(LSTM(device=device, dropout=0.1, emb_tech=emb_tech, input_shape=input_shape), input=data, output=rating, device=device, useTitle=useTitle)
    elif key == '2':
        train(BiLSTM(device=device, dropout=0.1, emb_tech=emb_tech, input_shape=input_shape), input=data, output=rating, device=device, useTitle=useTitle)      
    elif key == '3':
        train(XGBoost(emb_tech=emb_tech, useTitle=useTitle), input=data, output=rating, device=device, useTitle=useTitle)      
    elif key == '4':
        train(LR(emb_tech=emb_tech, useTitle=useTitle), input=data, output=rating, device=device, useTitle=useTitle)
    elif key == '5':
        train(CNNnLSTM(device=device, input_shape=input_shape, useTitle=useTitle, emb_tech=emb_tech, dropout=0.1), input=data, output=rating, device=device, useTitle=useTitle)
    elif key == '6':
        train(CNNnBiLSTM(device=device, input_shape=input_shape, useTitle=useTitle, emb_tech=emb_tech, dropout=0.1), input=data, output=rating, device=device, useTitle=useTitle)
    elif key == '7':
        train(GRU(device=device, input_shape=input_shape, emb_tech=emb_tech, dropout=0.1), input=data, output=rating, device=device, useTitle=useTitle)
    elif key == '8':
        train(BiGRU(device=device, input_shape=input_shape, emb_tech=emb_tech, dropout=0.1), input=data, output=rating, device=device, useTitle=useTitle)
    elif key == '9':
        train(Transformer(device=device, input_shape=input_shape, emb_tech=emb_tech, dropout=0.1), input=data, output=rating, device=device, useTitle=useTitle)
    elif key == '10':
        train(CNN(device=device, input_shape=input_shape, emb_tech=emb_tech, dropout=0.1), input=data, output=rating, device=device, useTitle=useTitle)
    else:
        print('Wrong key of model, please choose again')

def train(model, input, output, device, useTitle):
    model.to(device)
    direction = 'with_title' if useTitle else 'no_title'
    model_direction = 'phobert' if model.emb_tech == 1 else 'phow2v'
    ML_model = ['XGBoost', 'Logistic_Regression']

    # Splitting dataset
    train_size = int(0.8*input.size(0))
    val_size = int(0.1*train_size)
    train_size -= val_size
    test_size = input.size(0) - train_size - val_size

    if model.model_name in ML_model:
        _, output = torch.max(output, 1)
        output = output.numpy()
        train_size += val_size
        train_data = input[:train_size].cpu().numpy()
        train_label = output[:train_size]
        test_data = input[train_size:].cpu().numpy()
        test_label = output[train_size:]

        print("Training")
        model.train()
        model(train_data, train_label)

        model.eval()
        pred = model(x=test_data, train=False)

        report = classification_report(test_label, pred)
        print(report)

    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = opt.Adam(model.parameters(), lr=0.000001)
        scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, patience=25, factor=0.4, verbose=True, mode='min')

        # dataset = TensorDataset(input, output)
        train_data = DataLoader(TensorDataset(input[:train_size], output[:train_size]), batch_size=batch_size, shuffle=True)
        valid_data = DataLoader(TensorDataset(input[train_size:train_size+val_size], output[train_size:train_size+val_size]), batch_size=batch_size, shuffle=True)
        test_data = DataLoader(TensorDataset(input[-1*test_size:], output[-1*test_size:]), batch_size=batch_size, shuffle=True)

        best_acc = 0
        best_loss = 10000
        val_train_history = {
            'accuracy': [],
            'epoch': []
        }

        for epoch in range(num_epoch):
            train_bar = tqdm(train_data, desc=f"Epoch {epoch+1}/{num_epoch}:")
            val_bar = tqdm(valid_data, desc=f"Epoch {epoch + 1}/{num_epoch}:")

            model.train()
            for batch in train_bar:
                input_ids = batch[0].to(device)
                true_output = batch[1].float().to(device)
            
                pred_output = model(input_ids)
                loss = criterion(pred_output, true_output)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_bar.set_postfix(loss=loss.item())

            # Validation
            model.eval()
            total_val_loss = 0
            total = 0
            correct = 0

            for batch in val_bar:
                input_ids = batch[0].to(device)
                true_output = batch[1].float().to(device)
            
                pred_output = model(input_ids)
                loss = criterion(pred_output, true_output)

                total_val_loss += loss.item()

                _, predicted = torch.max(pred_output.data, 1)
                _, label = torch.max(true_output, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            avg_val_loss = total_val_loss / len(valid_data)
            accuracy = correct / total

            print(f'Epoch {epoch + 1}/{num_epoch}: Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')

            val_train_history['accuracy'].append(accuracy)
            val_train_history['epoch'].append(epoch + 1)

            # Compare current model with previous model
            if best_acc < accuracy or (best_acc == accuracy and best_loss > avg_val_loss):
                torch.save(model.state_dict(), f'res/models/{direction}/{model_direction}/{model.model_name}.pth')
                best_acc = accuracy
                best_loss = avg_val_loss
                print('Model saved locally, current accuracy:', best_acc)

            scheduler.step(avg_val_loss)

        # Test model performance
        test_bar = tqdm(test_data, desc=f"Epoch {epoch + 1}/{num_epoch}:")
        model.load_state_dict(torch.load(f'res/models/{direction}/{model_direction}/{model.model_name}.pth'))
        # Save model to Drive
        torch.save(model.state_dict(), f'{DRIVE_PATH}/models/{direction}/{model_direction}/{model.model_name}.pth')
        print(f'MODEL saved - {DRIVE_PATH}/models/{direction}/{model_direction}/{model.model_name}.pth')

        model.eval()

        predicted = []
        label = []
        for batch in test_bar:
            input_ids = batch[0].to(device)
            true_output = batch[1].float().to(device)
        
            pred_output = model(input_ids)
            loss = criterion(pred_output, true_output)

            total_val_loss += loss.item()

            _, temp_predicted = torch.max(pred_output.data, 1)
            _, temp_label = torch.max(true_output, 1)
            temp_predicted = temp_predicted.cpu().numpy()
            temp_label = temp_label.cpu().numpy()

            predicted.append(temp_predicted)
            label.append(temp_label)

        predicted = np.array(predicted[0])
        label = np.array(label[0])
        report = classification_report(label, predicted)
        print(report)

    # Save report to Drive
    with open(f'{DRIVE_PATH}/report/{direction}/{model_direction}/{model.model_name}.txt', 'w') as file:
        file.write(report)
    print(f'REPORT saved - {DRIVE_PATH}/report/{direction}/{model_direction}/{model.model_name}.txt')

    # Visualize model val train processing
    plt.figure()
    plt.plot(val_train_history['accuracy'], label='Valid Accuracy')
    plt.title(f'{model.model_name} Training Process')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    # Save image to Drive
    plt.savefig(f'{DRIVE_PATH}/train_process/{direction}/{model_direction}/{model.model_name}.png')
    plt.close()

    print(f'Image saved - {DRIVE_PATH}/train_process/{direction}/{model_direction}/{model.model_name}.png')
