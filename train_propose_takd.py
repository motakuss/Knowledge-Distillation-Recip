import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import random_split,DataLoader
from src.model import TeacherModel, StudentModel, TeacherAssistantModel
import torch.optim as optimizers
from src.kd_loss.fitnet import HintLearningLoss
from src.kd_loss.st import SoftTargetLoss
from src.utils import EarlyStopping
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pickle

'''
    FitNets: Hint for Thin Deep Nets
'''

def main():
    for i in range(20):
        print(i)
        torch.manual_seed(i)
        np.random.seed(i)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        data_dir = './data/cifar10'
        transform = transforms.Compose([transforms.ToTensor()
                                        ,transforms.Normalize(mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        
        cifar10_train = datasets.CIFAR10(root=data_dir,
                                        download=True,
                                        train=True,
                                        transform=transform)
        cifar10_test = datasets.CIFAR10(root=data_dir,
                                        download=True,
                                        train=False,
                                        transform=transform)
        
        n_samples = len(cifar10_train)
        n_train = int(n_samples * 0.8)
        n_val = n_samples - n_train
        cifar10_train, cifar10_val = random_split(cifar10_train, [n_train, n_val])
        
        train_dataloader = DataLoader(cifar10_train,
                                    batch_size=128,
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=8)
        val_dataloader = DataLoader(cifar10_val,
                                    batch_size=128,
                                    shuffle=False)
        test_dataloader = DataLoader(cifar10_test,
                                    batch_size=128,
                                    shuffle=False)
        
        teacher = TeacherModel().to(device)
        ta = TeacherAssistantModel().to(device)
        student = StudentModel().to(device)
        
        teacher.load_state_dict(torch.load('./logs/train_teacher/teacher_param' + str(i) + '.pth'))
        
        regressor_ta = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True) 
        ).to(device)
        
        regressor_student = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True) 
        ).to(device)
        
        #todo: optimization ta with fitnet training
        #todo: Fitnets Training Stage1: Train guided layers with teacher hints
        loss_hint = HintLearningLoss()
        hint_epochs = 100
        loss_score = float('inf')
        ta_optim = optimizers.Adam(list(ta.parameters()) + list(regressor_ta.parameters()))
        ta.train()
        regressor_ta.train()
        for epoch in range(hint_epochs):
            hint_loss = 0.
            val_hint_loss = 0.
            for (x,t) in tqdm(train_dataloader, leave = False):
                x, t = x.to(device), t.to(device)
                #todo: 教師モデルから中間層出力を得る
                teacher.eval()
                with torch.no_grad():
                    teacher_features = teacher.extract_features(x)
                #todo: taモデルにguide layerつける
                #todo: taモデルから中間層出力を得る
                ta_features = ta.extract_features(x)
                ta_guide = regressor_ta(ta_features)
                loss = loss_hint(teacher_features, ta_guide)
                ta_optim.zero_grad()
                loss.backward()
                ta_optim.step()
                hint_loss += loss.item()
            hint_loss /= len(train_dataloader)
            
            # calculate validation loss
            teacher.eval()
            ta.eval()
            regressor_ta.eval()
            with torch.no_grad():
                for (x,t) in val_dataloader:
                    x, t = x.to(device), t.to(device)
                    teacher_features = teacher.extract_features(x)
                    ta_features = ta.extract_features(x)
                    student_guide = regressor_ta(ta_features)
                    loss = loss_hint(student_guide, teacher_features)
                    val_hint_loss += loss.item()
            val_hint_loss /= len(val_dataloader)
            if loss_score >= val_hint_loss:
                print('test')
                loss_score = val_hint_loss
                torch.save(ta.state_dict(), './logs/train_propose_takd/ta_middle_param' + str(i) + '.pth')  
            
            print(f'epoch: {epoch+1}, hint_loss: {hint_loss:.3f}, val_hint_loss: {val_hint_loss:.3f}')
        
        #todo: second training(soft target) teacher and ta 
        ta_optim = optimizers.Adam(ta.parameters())
        loss_fn = nn.CrossEntropyLoss()
        soft_loss = SoftTargetLoss()
        ta.load_state_dict(torch.load('./logs/train_propose_takd/ta_middle_param' + str(i) + '.pth'))
        T = 10
        epochs = 100
        score = 0.
        #es = EarlyStopping(patience=10, verbose=1)
        ta_hist = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        for epoch in range(epochs):
            train_loss = 0.
            train_acc = 0.
            val_acc = 0.
            val_loss = 0.
            ta.train()
            teacher.eval()
            for (x,t) in train_dataloader:
                x, t = x.to(device), t.to(device)
                preds = ta(x)
                with torch.no_grad():
                    targets = teacher(x)
                loss = loss_fn(preds, t) + T * T * soft_loss(preds, targets)
                ta_optim.zero_grad()
                loss.backward()
                ta_optim.step()
                train_loss += loss.item()
                train_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)
            
            ta.eval()
            teacher.eval()
            with torch.no_grad():
                for (x,t) in val_dataloader:
                    x, t = x.to(device), t.to(device)
                    preds = ta(x)
                    targets = teacher(x)
                    loss = loss_fn(preds, t) + T * T * soft_loss(preds, targets)
                    val_loss += loss.item()
                    val_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
                val_loss /= len(val_dataloader)
                val_acc /= len(val_dataloader)
            
            ta_hist['loss'].append(train_loss)
            ta_hist['accuracy'].append(train_acc)
            ta_hist['val_loss'].append(val_loss)
            ta_hist['val_accuracy'].append(val_acc)
            
            print(f'epoch: {epoch+1}, loss: {train_loss:.3f}, accuracy: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_accuracy: {val_acc:.3f}')
            #if es(val_loss):
            #    break
            # 一番良いパラメータの選定
            if score <= val_acc:
                print('test')
                score = val_acc
                torch.save(ta.state_dict(), './logs/train_propose_takd/ta_param' + str(i) + '.pth')    
        """    
        #todo: distillation ta model test
        ta.eval()
        test_loss = 0.
        test_acc = 0.
        with torch.no_grad():
            for (x,t) in test_dataloader:
                x, t = x.to(device), t.to(device)
                preds = ta(x)
                loss = loss_fn(preds, t)
                test_loss += loss.item()
                test_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
            print(f'test_loss: {test_loss:.3f}, test_accuracy: {test_acc:.3f}')
        """
        
        # student training by optimized TA model
        #todo: Fitnets Training Stage1: Train guided layers with TA hint
        loss_score = float('inf')
        student_optim = optimizers.Adam(list(student.parameters()) + list(regressor_student.parameters()))
        #es = EarlyStopping(patience=10, verbose=1)
        student_hist = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        # 最適なTAのload
        ta.load_state_dict(torch.load('./logs/train_propose_takd/ta_param' + str(i) + '.pth'))
        # 学習
        for epoch in range(hint_epochs):
            hint_loss = 0.
            val_hint_loss = 0.
            student.train()
            ta.eval()
            regressor_student.train()
            for (x,t) in train_dataloader:
                x, t = x.to(device), t.to(device)
                with torch.no_grad():
                    ta_features = ta.extract_features(x)
                student_features = student.extract_features(x)
                student_guide = regressor_student(student_features)
                loss = loss_hint(student_guide, ta_features)
                student_optim.zero_grad()
                loss.backward()
                student_optim.step()
                hint_loss += loss.item()
            hint_loss /= len(train_dataloader)
            
            # validation
            ta.eval()
            student.eval()
            regressor_student.eval()
            with torch.no_grad():
                for (x, t) in val_dataloader:
                    x, t = x.to(device), t.to(device)
                    ta_features = ta.extract_features(x)
                    student_features = student.extract_features(x)
                    student_guide = regressor_student(student_features)
                    loss = loss_hint(student_guide, ta_features)
                    val_hint_loss += loss.item()
            val_hint_loss /= len(val_dataloader)
            
            if loss_score >= val_hint_loss:
                print('test')
                loss_score = val_hint_loss
                torch.save(student.state_dict(), './logs/train_propose_takd/student_middle_param' + str(i) + '.pth')  
            
            print(f'epoch: {epoch+1}, hint_loss: {hint_loss:.3f}, val_hint_loss: {val_hint_loss:.3f}')
            
        student_optim = optimizers.Adam(student.parameters())
        epoch = 100
        T = 10
        score = 0.
        student.load_state_dict(torch.load('./logs/train_propose_takd/student_middle_param' + str(i) + '.pth'))
        for epoch in range(epochs):
            train_loss = 0.
            train_acc = 0.
            val_loss = 0.
            val_acc = 0.
            student.train()
            ta.eval()
            for (x,t) in train_dataloader:
                x,t = x.to(device), t.to(device)
                preds = student(x)
                with torch.no_grad():
                    targets = ta(x)
                loss = loss_fn(preds, t) + T * T * soft_loss(preds, targets)
                student_optim.zero_grad()
                loss.backward()
                student_optim.step()
                train_loss += loss.item()
                train_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
            train_acc /= len(train_dataloader)
            train_loss /= len(train_dataloader)
            
            # student validation
            student.eval()
            ta.eval()
            with torch.no_grad():
                for (x,t) in val_dataloader:
                    x,t = x.to(device), t.to(device)
                    preds = student(x)
                    targets = ta(x)
                    loss = loss_fn(preds, t) +  T * T * soft_loss(preds, targets)
                    val_loss += loss.item()
                    val_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
                val_loss /= len(val_dataloader)
                val_acc /= len(val_dataloader)
            
            student_hist['loss'].append(train_loss)
            student_hist['accuracy'].append(train_acc)
            student_hist['val_loss'].append(val_loss)
            student_hist['val_accuracy'].append(val_acc)

            print(f'epoch: {epoch+1}, loss: {train_loss:.3f}, accuracy: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_accuracy: {val_acc:.3f}')
            
            # 一番良いパラメータの選定
            if score <= val_acc:
                print('test')
                score = val_acc
                torch.save(student.state_dict(), './logs/train_propose_takd/student_param' + str(i) + '.pth')
            #if es(val_loss):
            #    break
            
        with open('./logs/train_propose_takd/hist'+str(i)+'.pickle', mode='wb') as f:
                pickle.dump(student_hist, f)
                
        # distillation student test
        student.load_state_dict(torch.load('./logs/train_propose_takd/student_param' + str(i) + '.pth'))
        student.eval()
        test_loss = 0.
        test_acc = 0.
        with torch.no_grad():
            for (x,t) in test_dataloader:
                x, t = x.to(device), t.to(device)
                preds = student(x)
                loss = loss_fn(preds, t)
                test_loss += loss.item()
                test_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        test = {'acc': [], 'loss': []}
        test['acc'].append(test_acc)
        test['loss'].append(test_loss)
        print(f'test_loss: {test_loss:.3f}, test_accuracy: {test_acc:.3f}')
        # test_loss: 0.787, test_accuracy: 0.826
        # store data
        # train loss,accをepochごとに保存
        # test acc,lossを保存
        with open('./logs/train_propose_takd/test' + str(i) + '.pickle', mode='wb') as f:
                pickle.dump(test, f)
    
        
if __name__ == '__main__':
    main()
    
    
    
        
        