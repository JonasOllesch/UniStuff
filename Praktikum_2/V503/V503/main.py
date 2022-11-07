import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import class_structure
from sklearn.model_selection import train_test_split

#Signal=1=positive
#Background=0=negative
#tp/(tp+fp)
def precision(truelabels,predictedlabels):
    return (len(predictedlabels[(truelabels==1) & (predictedlabels==1)]))/(len(predictedlabels[(truelabels==1) & (predictedlabels==1)])+len(predictedlabels[(truelabels==0) & (predictedlabels==1)]))
#tp/(tp+fn)
def recall(truelabels,predictedlabels):
    return (len(predictedlabels[(truelabels==1) & (predictedlabels==1)]))/(len(predictedlabels[(truelabels==1) & (predictedlabels==1)])+len(predictedlabels[(truelabels==1) & (predictedlabels==0)]))
#fn/(fn+tp)
def significance(truelabels,predictedlabels):
    return len(predictedlabels[(truelabels==1) & (predictedlabels==0)])/(len(predictedlabels[(truelabels==1) & (predictedlabels==0)])+len(predictedlabels[(truelabels==1) & (predictedlabels==1)]))


#Create data with labels 
#0=Background
#1=Signal
print("Loading Data")
Signal=pd.read_hdf('NeutrinoMC.hdf5','Signal')
Signal=(Signal[~np.isnan(Signal['x'])].to_numpy())[:,2:5]
ysignal=np.ones(len(Signal))
Background=(pd.read_hdf('NeutrinoMC.hdf5','Background').to_numpy())[0:25000,0:3]
ybackground=np.zeros(len(Background))

y=np.concatenate([ysignal,ybackground])
y=np.array(y,dtype=int)

X=np.concatenate([Signal,Background],axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=5000, random_state=42)

numberOfTestBackground=20000
numberOfTestSignals=10000
testBackground=X_test[y_test==0][0:numberOfTestBackground]
testSignal=X_test[y_test==1][0:numberOfTestSignals]
testinput=np.concatenate([testBackground,testSignal],axis=0)
testinputlabels=np.concatenate((np.zeros(numberOfTestBackground),np.ones(numberOfTestSignals)))

#KNN and prediction
print("Invoke KNN with prediction")

#Prediction for k=10
classifier=class_structure.KNN(10)
classifier.fit(X_train,y_train)
print("startprediction")
predictions=classifier.predict(testinput)
print("endprediction")

#Prediction for log
testinputlog=testinput
X_trainlog=X_train
testinputlog[:,0]=np.log10(testinput[:,0].astype(float))
X_trainlog[:,0]=np.log10(X_train[:,0].astype(float))

classifierlog=class_structure.KNN(10)
classifierlog.fit(X_trainlog,y_train)
print("Starts log predictions")
predictionslog=classifier.predict(testinputlog)
print("end log predictions")

#prediction for k=20
classifier20=class_structure.KNN(20)
classifier20.fit(X_train,y_train)
print("startpredictionk=20")
predictions20=classifier20.predict(testinput)
print("endpredictionk=20")

#Calculate Metrics
file = open("values.txt", "w")
file.write("k=10: "+"\n")
file.write("Precision: "+str(precision(testinputlabels,predictions))+"\n")
file.write("Recall: "+str(recall(testinputlabels,predictions))+"\n")
file.write("Significance: "+str(significance(testinputlabels,predictions))+"\n")
file.write("\n")
file.write("log: "+"\n")
file.write("Precision: "+str(precision(testinputlabels,predictionslog))+"\n")
file.write("Recall: "+str(recall(testinputlabels,predictionslog))+"\n")
file.write("Significance: "+str(significance(testinputlabels,predictionslog))+"\n")
file.write("\n")
file.write("k=20: "+"\n")
file.write("Precision: "+str(precision(testinputlabels,predictions20))+"\n")
file.write("Recall: "+str(recall(testinputlabels,predictions20))+"\n")
file.write("Significance: "+str(significance(testinputlabels,predictions20))+"\n")
file.write("\n")

#Plotting
xdim=1
ydim=2
print("startplotting")

plt.plot(testBackground[:,xdim],testBackground[:,ydim],'x',label='true0')
plt.plot(testSignal[:,xdim],testSignal[:,ydim],'x',label='true1')
plt.legend()
plt.savefig('build/true.pdf')

plt.figure()
plt.plot(testinput[predictions==0][:,xdim],testinput[predictions==0][:,ydim],'x',label='predicted0')
plt.plot(testinput[predictions==1][:,xdim],testinput[predictions==1][:,ydim],'x',label='predicted1')
# plt.plot(X_train[y_train==0][:,xdim],X_train[y_train==0][:,ydim],'x',markersize=0.5,label='train0')
# plt.plot(X_train[y_train==1][:,xdim],X_train[y_train==1][:,ydim],'x',markersize=0.5,label='train1')
plt.legend()
plt.savefig('build/predicted.pdf')

plt.figure()
plt.plot(testinput[predictions20==0][:,xdim],testinput[predictions20==0][:,ydim],'x',label='predicted20_0')
plt.plot(testinput[predictions20==1][:,xdim],testinput[predictions20==1][:,ydim],'x',label='predicted20_1')
# plt.plot(X_train[y_train==0][:,xdim],X_train[y_train==0][:,ydim],'x',markersize=0.5,label='train0')
# plt.plot(X_train[y_train==1][:,xdim],X_train[y_train==1][:,ydim],'x',markersize=0.5,label='train1')
plt.legend()
plt.savefig('build/predicted20.pdf')

plt.figure()
plt.plot(testinputlog[predictionslog==0][:,xdim],testinputlog[predictionslog==0][:,ydim],'x',label='predictedlog0')
plt.plot(testinputlog[predictionslog==1][:,xdim],testinputlog[predictionslog==1][:,ydim],'x',label='predictedlog1')
# plt.plot(X_train[y_train==0][:,xdim],X_train[y_train==0][:,ydim],'x',markersize=0.5,label='train0')
# plt.plot(X_train[y_train==1][:,xdim],X_train[y_train==1][:,ydim],'x',markersize=0.5,label='train1')
plt.legend()
plt.savefig('build/predictedlog.pdf')







