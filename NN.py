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



# frage ob CUDA möglich ist und gib es als Wahrheitsvariable aus
use_cuda = torch.cuda.is_available()

# wenn CUDA möglich ist, definiere, dass das Gerät <device> der ersten Grafikkarte zugeordnet wird
device   = torch.device('cuda:0' if use_cuda else 'cpu')

# falls über die CPU gerechnte wird, nutze n CPU-Kerne
if use_cuda == False:
    device   = torch.device('cpu')
    torch.set_num_threads(15)

X = []
Y = []
with open('./balanced_t_data.csv', 'r') as fin:
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

#exit()

X = torch.tensor(X)
Y = torch.tensor(Y)
        


# splitte die Trainingsdaten zu je (ca.) 50 % in tatsächlichen Trainingsdaten und Validierungsdaten

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

X_train = X_train.to(device) 
X_test  = X_test.to(device) 
Y_train = Y_train.to(device) 
Y_test  = Y_test.to(device) 



# Diverse Vorgaben zum Neuronalen Netz
Number_Input_Neurons  = len(X[0])                # Anzahl der Eingangsneuronen angepasst an die Größe der Trainingsdaten
lr                    = 0.0001                    # Lernrate (bei Verwendung des ADAM-Optimizers sollte diese locker unter 1e-5 sein)
N_epoch               = 4000                      # Anzahl der Durchläufe
Model_Parameter_File  = "model_status_cuda.pt"   # Dateiname für die Weights




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

# Loss Function vorgeben (Mean Squared Error Loss)
loss_fun = torch.nn.MSELoss()

# Optimizer vorgeben (SGD Stochastic Gradient Descent)  # hat sich für dieses Model als signifikant 
# schlecht gegenüber dem ADAM-Optimizer herausgestellt
# optimizer = torch.optim.SGD(model.parameters(), lr = Learning_Rate, momentum=0.7)

# ADAM Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Optimierungsprozess des NN


LOSS_train = []
LOSS_test  = []

# Iteriere über die vorgegebene Anzahl an Epochen
for epoch in range(N_epoch):

    # do zero the gradient
    optimizer.zero_grad()
    
    
    # Vorwärts Propagierung
    Y_pred = model(X_train).to(device) 

    # Berechne Fehler und gebe ihn aus
    loss      = loss_fun(Y_pred, Y_train)
    loss_test = loss_fun(model(X_test).to(device), Y_test)
    
    
    #print('epoch: ', epoch, ' loss: ', loss.item())
    print(80*'~')
    print('#Iteration :', epoch)
    print('loss value :', float(loss))

    # 
    LOSS_train.append(float(loss))
    LOSS_test.append(float(loss_test))
    


    # Führe Modellfehler rückwertig auf das Modell zurück
    loss.backward()

    # Update the parameters
    optimizer.step()






# Plotte Ergebnisse zur Validierung des Modells

plt.plot(LOSS_train, color='green', label='training')
plt.plot(LOSS_test, color='blue', label='test')

plt.title('Model Optimization')
plt.ylabel('Loss')
plt.xlabel('Number of Optimization Steps')

plt.legend()
plt.show()

torch.save(model.state_dict(), Model_Parameter_File)
