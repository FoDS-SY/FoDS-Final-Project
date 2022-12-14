#!/usr/bin/env python
# coding: utf-8

# Find the direction of dataset
import numpy as np
import pandas as pd
import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import random # for visualization
print('Libraries Imported')
import joblib


from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import numpy as np
import cv2


# ## Preprocessing

#path = '../input/mias-mammography/all-mias/'
path = './all-mias/'

print("reading dataframe")
#info=pd.read_csv("../input/mias-mammography/Info.txt",sep=" ")
info=pd.read_csv("./Info.txt",sep=" ")
info=info.drop('Unnamed: 7',axis=1)

info.dropna(subset = ["SEVERITY"], inplace=True)
info.reset_index(inplace = True)
#info

info = info.drop([3], axis=0)
info.reset_index(inplace = True)
#info

# taking the images filenames in to dictionary
ids = {}
for i in range(len(info)):
    ids[i] = info.REFNUM[i]
#ids

# Turning our outputs B-M to 1-0
label = []
num_B = 0
num_M = 0 
for i in range(len(info)):
    if info.SEVERITY[i] == 'B':
        label.append(1)
        num_B+=1
    else:
        label.append(0)
        num_M+=1 

label = np.array(label)

label.shape,num_B,num_M

num_list = [num_B,num_M]

name_list = ['Benign','Malignant']
plt.bar(range(len(num_list)), num_list,tick_label=name_list)
plt.show()


# define the every images filepaths in to list
img_name = []

for i in range(len(label)):
        img_name.append(path + info.REFNUM[i]+ '.pgm')

img_name = np.array(img_name)

#print(img_name)
print(f'image addres amount {img_name.shape}')

img_path = []
last_label = []

#for i in range(len(img_name)):
img_change_1 = []
for i in range(122):
    img = cv2.imread(img_name[i], 0)
    img = cv2.resize(img, (224,224))
    img_path.append(img)
    last_label.append(label[i])
    rows, cols= img.shape
    for angle in range(10):
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)    #Rotate 0 degree
            img_rotated = cv2.warpAffine(img, M, (224, 224))
            img_path.append(img_rotated)
            img_change_1.append(img_rotated)
            if label[i] == 1:
                last_label.append(1)
            else:
                last_label.append(0)

img_path[1]


# view image random images
def view_25_random_image():
    fig = plt.figure(figsize = (15, 10))
    for i in range(25):
        #rand = random.randint(0,len(label))
        #rand = random.randint(0,3)
        ax = plt.subplot(5, 5, i+1)
    
        #img = cv2.imread(img_path[rand], 0)
        #img = cv2.resize(img, (256,256))
        img =img_change_1[i]
       
        plt.tight_layout()
        plt.axis('off')
        plt.imshow(img)
    fig.savefig('random_25_image_fig.png')

random_images = view_25_random_image()


# split train and test set
x_train, x_test, y_train, y_test = train_test_split(img_path, last_label, test_size = 0.2, random_state = 32)

len(x_train),len(x_test),len(y_train),len(y_test)

x_train = np.array(x_train)
x_test = np.array(x_test)

# flatten the images 
nsamples, nx, ny = x_train.shape
x_train2 = x_train.reshape((nsamples,nx*ny))
nsamples1, nx1, ny1 = x_test.shape
x_test2 = x_test.reshape((nsamples1,nx1*ny1))
x_train = np.reshape(x_train, (nsamples, nx, ny, 1)) # 1 for gray scale
x_test = np.reshape(x_test, (nsamples1, nx1, ny1, 1))
x_train.shape,x_train2.shape,x_test.shape,x_test2.shape

#(a,b,c)=x_train.shape # (35136, 224, 224)
#x_train = np.reshape(x_train, (a, b, c, 1)) # 1 for gray scale
#(a, b, c)=x_test.shape
#x_test = np.reshape(x_test, (a, b, c, 1))


# ## SVM code

from sklearn import svm
from sklearn.metrics import RocCurveDisplay
model=svm.SVC(kernel = "rbf",C=10)
model.fit(x_train2,y_train)

# save the model
joblib.dump(model, 'svm1.model')
# load the model
model = joblib.load('svm1.model')

# model=svm.SVC(kernel = "rbf")

y_pred=model.predict(x_test2)
# print("The predicted Data is :")
# print(y_pred)
# print("The actual data is:")
# print(np.array(y_test))
# print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")

accuracy_score(y_pred,y_test)
print(classification_report(y_pred,y_test))

confusion_matrix(y_pred,y_test)

cm = confusion_matrix(y_pred,y_test)
def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    
    plt.imshow(cm, interpolation='nearest')    
    plt.title(title)   
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)   
    plt.yticks(num_local, labels_name)   
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
plot_confusion_matrix(cm, name_list, "HAR Confusion Matrix")
# plt.savefig('/HAR_cm.png', format='png')
plt.show()


x_test.shape

# shuffle and split training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Learn to predict each class against the other
#classifier = OneVsRestClassifier(
    #svm.SVC(kernel="linear", probability=True, random_state=random_state)
#)
from sklearn.multiclass import OneVsRestClassifier
#model=svm.SVC(kernel = "rbf")
#y_score = model.fit(x_train2,y_train)
classifier = OneVsRestClassifier(
    svm.SVC(kernel="linear", probability=True))
y_score = classifier.fit(x_train2, y_train).decision_function(x_test2)

ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(classifier, x_test2, y_test, ax=ax, alpha=0.8)
plt.plot([0, 1], [0, 1], 'k--')
plt.show()


C_list = [0.00001,0.0001,0.001,0.01,0.1,1,10,100]
acc_list_svm = []
for i in range(len(C_list)):
    model=svm.SVC(kernel = "rbf",C = C_list[i])
    model.fit(x_train2,y_train)
    # accuracy
    y_pred=model.predict(x_test2)
    acc_temp = accuracy_score(y_pred,y_test)
    acc_list_svm.append(acc_temp)

plt.plot(C_list,acc_list_svm)
plt.xlabel("C in SVM")
plt.ylabel("accuracy")


# Plot ROC curve
#plt.clf()
#plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
#plt.plot([0, 1], [0, 1], 'k--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.0])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiverrating characteristic example')
#plt.legend(loc="lower right")
#plt.show()


#fpr_svm = fpr
#tpr_svm = tpr

#fpr_svm


# ## Random forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay
model=RandomForestClassifier()

model.fit(x_train2,y_train)

# save the model
joblib.dump(model, 'rf1.model')
# load the model
model = joblib.load('rf1.model')

y_pred=model.predict(x_test2)
#y_pred

rfc = RandomForestClassifier(n_estimators=5, random_state=42)
rfc.fit(x_train2, y_train)
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(rfc, x_test2, y_test, ax=ax, alpha=0.8)
plt.plot([0, 1], [0, 1], 'k--')
plt.show()

rfc_rf = rfc 

ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(rfc, x_test2, y_test, ax=ax, alpha=0.8)
plt.plot([0, 1], [0, 1], 'k--')
plt.show()


accuracy_score(y_pred,y_test)
print(classification_report(y_pred,y_test))

confusion_matrix(y_pred,y_test)

cm = confusion_matrix(y_pred,y_test)
def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    
    plt.imshow(cm, interpolation='nearest')   
    plt.title(title)  
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90) 
    plt.yticks(num_local, labels_name)   
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
plot_confusion_matrix(cm, name_list, "HAR Confusion Matrix")
# plt.savefig('/HAR_cm.png', format='png')
plt.show()


est_list = [1,2,3,4,5,6,7,8,9,10]
acc_list = []
for i in range(len(est_list)):
    model= RandomForestClassifier(n_estimators= est_list[i], random_state=42)
    model.fit(x_train2,y_train)
    # accuracy
    y_pred=model.predict(x_test2)
    acc_temp = accuracy_score(y_pred,y_test)
    acc_list.append(acc_temp)

plt.plot(est_list,acc_list)
plt.xlabel("number of estimators in Random Frest")
plt.ylabel("accuracy")


# ## Decision Tree

from sklearn.tree import DecisionTreeClassifier

rfc_dt = DecisionTreeClassifier()
rfc_dt.fit(x_train2, y_train)
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(rfc_dt, x_test2, y_test, ax=ax, alpha=0.8)
#svc_disp.plot(ax=ax, alpha=0.8)
plt.plot([0, 1], [0, 1], 'k--')
plt.show()

y_pred= rfc_dt.predict(x_test2)
#y_pred
dtc=DecisionTreeClassifier()

#dtc.fit(x_train2,y_train)
# save the model


# save the model
#joblib.dump(dtc, 'dct1.model')
# load the model
dtc = joblib.load('dct1.model')

y_pred_dtc=dtc.predict(x_test2)

y_pred_dtc

accuracy_score(y_pred_dtc,y_test)

accuracy_score(y_pred_dtc,y_test)
print(classification_report(y_pred_dtc,y_test))

cm = confusion_matrix(y_pred_dtc,y_test)
confusion_matrix(y_pred_dtc,y_test)

def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    
    plt.imshow(cm, interpolation='nearest')   
    plt.title(title)   
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)   
    plt.yticks(num_local, labels_name)    
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
plot_confusion_matrix(cm, name_list, "HAR Confusion Matrix")
# plt.savefig('/HAR_cm.png', format='png')
plt.show()


# ##  A Naive Bayes classifier

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import RocCurveDisplay
nb=GaussianNB()
nb.fit(x_train2,y_train)
y_pred_nb=nb.predict(x_test2)

var_smoothing_list = np.logspace(0,-9, num=10)
var_smoothing_list

var_smoothing_list = np.logspace(0,-9, num=10)
acc_list_nb = []
for i in range(len(var_smoothing_list)):
    model= GaussianNB(var_smoothing= var_smoothing_list[i])
    model.fit(x_train2,y_train)
    # accuracy
    y_pred=model.predict(x_test2)
    acc_temp = accuracy_score(y_pred,y_test)
    acc_list_nb.append(acc_temp)
plt.plot(var_smoothing_list,acc_list_nb)
plt.xlabel("number of estimators in Naive Bayes")
plt.ylabel("accuracy")

acc_list_nb


from sklearn.model_selection import GridSearchCV
nb = GaussianNB()
params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
gs_NB = GridSearchCV(estimator=nb, 
                 param_grid=params_NB,   
                 verbose=1, 
                 scoring='accuracy') 
gs_NB.fit(x_train2, y_train)

gs_NB.best_params_


joblib.dump(gs_NB, 'gs1.model')
# load the model
gs_NB = joblib.load('gs1.model')

y_pred_nb = gs_NB.predict(x_test2)

ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(gs_NB, x_test2, y_test, ax=ax, alpha=0.8)
plt.plot([0, 1], [0, 1], 'k--')
plt.show()

accuracy_score(y_pred_nb,y_test)

print(classification_report(y_pred_nb,y_test))

confusion_matrix(y_pred_nb,y_test)
cm  = confusion_matrix(y_pred_nb,y_test)

def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    
    plt.imshow(cm, interpolation='nearest')   
    plt.title(title)    
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    
    plt.yticks(num_local, labels_name)    
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
plot_confusion_matrix(cm, name_list, "HAR Confusion Matrix")
# plt.savefig('/HAR_cm.png', format='png')
plt.show()


fig, ax = plt.subplots()
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(rfc, x_test2, y_test, ax=ax, alpha=0.8)
rfc_disp = RocCurveDisplay.from_estimator(classifier, x_test2, y_test, ax=ax, alpha=0.8)
rfc_disp = RocCurveDisplay.from_estimator(rfc, x_test2, y_test, ax=ax, alpha=0.8)
rfc_disp = RocCurveDisplay.from_estimator(rfc_dt, x_test2, y_test, ax=ax, alpha=0.8)
#rfc_disp = RocCurveDisplay.from_estimator(gs_NB, x_test2, y_test, ax=ax, alpha=0.8)

#plt.plot([0, 1], [0, 1], 'k--')
plt.show()
ax.legend(loc = 0, prop = {'size':2}) 
plt.show() 


# ## Logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay

clf = LogisticRegression(penalty='none', 
                         tol=0.1, solver='saga',
                         multi_class='multinomial').fit(x_train2,y_train)
rfc_lr = clf
rfc_lr.fit(x_train2, y_train)
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(rfc_lr, x_test2, y_test, ax=ax, alpha=0.8)
#svc_disp.plot(ax=ax, alpha=0.8)
plt.plot([0, 1], [0, 1], 'k--')
plt.show()

joblib.dump(rfc_lr, 'lr1.model')
# load the model
model = joblib.load('lr1.model')

#clf.fit(x_train2,y_train)

y_pred_dtc=clf.predict(x_test2)
y_pred_dtc

accuracy_score(y_pred_dtc,y_test)

print(classification_report(y_pred_dtc,y_test))

confusion_matrix(y_pred,y_test)

rfc = clf
rfc.fit(x_train2, y_train)
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(rfc, x_test2, y_test, ax=ax, alpha=0.8)
#svc_disp.plot(ax=ax, alpha=0.8)
plt.plot([0, 1], [0, 1], 'k--')
plt.show()

y_pred = y_pred_dtc
cm = confusion_matrix(y_pred,y_test)

def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    
    plt.imshow(cm, interpolation='nearest')    
    plt.title(title)    
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    
    plt.yticks(num_local, labels_name)  
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
plot_confusion_matrix(cm, name_list, "HAR Confusion Matrix")
# plt.savefig('/HAR_cm.png', format='png')
plt.show()


# # plot all the roc in one figure without cnn

fig, ax = plt.subplots() 
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(rfc, x_test2, y_test, ax=ax, alpha=0.8)
rfc_disp = RocCurveDisplay.from_estimator(classifier, x_test2, y_test, ax=ax, alpha=0.8)
rfc_disp = RocCurveDisplay.from_estimator(rfc_rf, x_test2, y_test, ax=ax, alpha=0.8)
rfc_disp = RocCurveDisplay.from_estimator(rfc_dt, x_test2, y_test, ax=ax, alpha=0.8)
rfc_disp = RocCurveDisplay.from_estimator(gs_NB, x_test2, y_test, ax=ax, alpha=0.8)
rfc_disp = RocCurveDisplay.from_estimator(rfc_lr, x_test2, y_test, ax=ax, alpha=0.8)
plt.plot([0, 1], [0, 1], 'k--')
plt.show()
ax.legend() 
plt.show() 


# ## CNN code

def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(224, 224, 1)))
    model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
  
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

model = create_model()
model.summary()

early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=0,restore_best_weights=True, verbose=1)

check_point_filepath = './'

model_check_point = ModelCheckpoint(filepath =check_point_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                    save_weights_only=False, mode='auto', save_freq='epoch')

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

load_model = True
if load_model:
    hist  = model.load_weights('m5.h5')
else:

    hist = model.fit(x_train,
                     y_train,
                     validation_split=0.2,
                     epochs=1000,
                     batch_size=64)

# save the model 
#model.save('m2.h5')
# model.save_weights('m5.h5')

loss_value , accuracy = model.evaluate(x_test, y_test)

print('Test_loss_value = ' +str(loss_value))
print('test_accuracy = ' + str(accuracy))

print(model.predict(x_test))

def Visualize_Result(acc,val_acc,loss, val_loss):
    fig, (ax1, ax2) = plt.subplots(nrows = 1,
                                   ncols = 2,
                                   figsize = (15,6),
                                   sharex =True)

    plot1 = ax1.plot(range(0, len(acc)),
                     acc,
                     label = 'accuracy')

    plot2 = ax1.plot(range(0, len(val_acc)),
                     val_acc,
                     label = 'val_accuracy')

    ax1.set(title = 'Accuracy And Val Accuracy progress',
            xlabel = 'epoch',
            ylabel = 'accuracy/ validation accuracy')

    ax1.legend()

    plot3 = ax2.plot(range(0, len(loss)),
                     loss,
                     label = 'loss')
    
    plot4 = ax2.plot(range(0, len(val_loss)),
                     val_loss,
                     label = 'val_loss')
    
    ax2.set(title = 'Loss And Val loss progress',
            xlabel = 'epoch',
            ylabel = 'loss/ validation loss')

    ax2.legend()

    fig.suptitle('Result Of Model', fontsize = 20, fontweight = 'bold')
    fig.savefig('Accuracy_Loss_figure.png')
    plt.tight_layout()
    plt.show()

visualize_result = Visualize_Result(hist.history['accuracy'],hist.history['val_accuracy'], hist.history['loss'], hist.history['val_loss'])


y_pred=model.predict(x_test)
y_pred.shape
accuracy_score(y_pred,y_test)

# calculating ROC
for  i in range(len(y_pred)):
    if y_pred[i]>0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0  
cf_matrix = confusion_matrix(y_test, y_pred)
cf_matrix

y_pred = y_pred_dtc
cm = confusion_matrix(y_test, y_pred)
def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    
    plt.imshow(cm, interpolation='nearest')    
    plt.title(title)    
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    
    plt.yticks(num_local, labels_name)    
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
plot_confusion_matrix(cm, name_list, "HAR Confusion Matrix")
# plt.savefig('/HAR_cm.png', format='png')
plt.show()

from sklearn.metrics import roc_curve, auc
nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = roc_curve(y_test, y_pred)
auc_keras = auc(nn_fpr_keras, nn_tpr_keras)
plt.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='Neural Network (Auc = %0.3f)' % auc_keras)
plt.plot([0, 1], [0, 1], 'k--')
plt.legend()
plt.show()


# ## ResNet code

class ResidualBlock(torch.nn.Module):
    def __init__(self,channels):
        super(ResidualBlock, self).__init__()
        self.channels=channels
        self.conv1=torch.nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2=torch.nn.Conv2d(channels,channels,kernel_size=3,padding=1)
    def forward(self,x):
        y=F.relu(self.conv1(x))
        y=self.conv2(y)
        return F.relu(x+y)
class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1=torch.nn.Conv2d(1,16,kernel_size=5)
        self.conv2=torch.nn.Conv2d(16,32,kernel_size=5)
        self.mp=torch.nn.MaxPool2d(2)
 
        self.rblock1=ResidualBlock(16)
        self.rblock2=ResidualBlock(32)
 
        self.fc=torch.nn.Linear(512,10)
    def forward(self,x):
        in_size=x.size(0)
        x=self.mp(F.relu(self.conv1(x)))
        x=self.rblock1(x)
        x=self.mp(F.relu(self.conv2(x)))
        x=self.rblock2(x)
        x=x.view(in_size,-1)
        x=self.fc(x)
        return x

model = Net2()

# Create optimizer
optimizer = optim.Adadelta(model.parameters(), lr=lr)

# Create a schedule for the optimizer
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

# Begin training and testing
for epoch in range(1, epochs + 1):
    train(model, train_loader, optimizer,epoch)
    test(model,  test_loader)
    scheduler.step()


# # plot all the roc in one figure with CNN

fig, ax = plt.subplots() 
ax = plt.gca()
#rfc_disp = RocCurveDisplay.from_estimator(rfc, x_test2, y_test, ax=ax, alpha=0.8)
rfc_disp = RocCurveDisplay.from_estimator(classifier, x_test2, y_test, ax=ax, alpha=0.8)
rfc_disp = RocCurveDisplay.from_estimator(rfc_rf, x_test2, y_test, ax=ax, alpha=0.8)
rfc_disp = RocCurveDisplay.from_estimator(rfc_dt, x_test2, y_test, ax=ax, alpha=0.8)
rfc_disp = RocCurveDisplay.from_estimator(gs_NB, x_test2, y_test, ax=ax, alpha=0.8)
rfc_disp = RocCurveDisplay.from_estimator(rfc_lr, x_test2, y_test, ax=ax, alpha=0.8)
plt.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='Neural Network (auc = %0.3f)' % auc_keras)
plt.plot([0, 1], [0, 1], 'k--')
ax.legend() 
plt.show() 

fig, ax = plt.subplots() 
ax = plt.gca()
#rfc_disp = RocCurveDisplay.from_estimator(rfc, x_test2, y_test, ax=ax, alpha=0.8)
rfc_disp = RocCurveDisplay.from_estimator(classifier, x_test2, y_test, ax=ax, alpha=0.8)
rfc_disp = RocCurveDisplay.from_estimator(rfc_rf, x_test2, y_test, ax=ax, alpha=0.8)
rfc_disp = RocCurveDisplay.from_estimator(rfc_dt, x_test2, y_test, ax=ax, alpha=0.8)
rfc_disp = RocCurveDisplay.from_estimator(gs_NB, x_test2, y_test, ax=ax, alpha=0.8)
rfc_disp = RocCurveDisplay.from_estimator(rfc_lr, x_test2, y_test, ax=ax, alpha=0.8)
plt.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='Neural Network (auc = %0.3f)' % auc_keras)
plt.plot([0, 1], [0, 1], 'k--')
ax.legend() 
plt.show() 

