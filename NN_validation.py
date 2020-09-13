import os.path
import torch
import torch.nn as nn
import numpy as np
import sys
import matplotlib.pyplot as plt
import time as time
import random as rand


from sklearn.model_selection import train_test_split

plt.style.use('ggplot')



device   = torch.device('cpu')



X = []
Y = []
with open('./balanced_v_data.csv', 'r') as fin:
    for line in fin.readlines():
        line = line.strip('\n').split(';')
        if line[-1] in ['1', '0']:
            X.append([float(val) for val in line[:-1]])
            Y.append([float(line[-1])])


ones  = 0
zeros = 0
for val in Y:
    if val[0] < 1:
        zeros += 1
    else:
        ones += 1
        
print('Anzahl an Daten            ', len(Y))
print('Anzahl an positiven Fällen ', ones)
print('Anzahl an negativen Fällen ', zeros)


        

# Diverse Vorgaben zum Neuronalen Netz
Number_Input_Neurons  = len(X[0])                
Model_Parameter_File  = "model_status_cuda.pt"   



class net(nn.Module):


    def __init__(self):
        super(net, self).__init__()
        
        self.lin1 = nn.Linear(Number_Input_Neurons, 128)
        self.lin2 = nn.Linear(128, 1)
        
    def forward(self, x):
    
        #x = torch.nn.functional.relu(self.lin1(x))
        x = self.lin1(x)
        x = torch.nn.functional.sigmoid(self.lin2(x))
                
        return x
 
 


    

# the neural net as object   
model = net().to(device)

# Lädt die gespeicherten Weights, wenn Datei vorhanden
if os.path.isfile(Model_Parameter_File):
    model.load_state_dict(torch.load(Model_Parameter_File))





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

acc_count = 0
for x, y in zip(X, Y):
    print(80*'x')
    print(x)
    print(y)
    x = torch.tensor(x).to(device)
    y_pred = int(round(float(model(x))))
    print('model_output:', y_pred)


    if y_pred == int(y[0]):
        acc_count += 1



print('Model accuracy is :', acc_count/len(X))







