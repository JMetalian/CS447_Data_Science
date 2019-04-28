import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import datasets

#Total playing time is not chosen due to a player can give a long break to 
#his/her playing career, then they can start playing again
relatedColumnsList=["LeagueIndex","Age","HoursPerWeek","APM","SelectByHotkeys",
                    "AssignToHotkeys","UniqueHotkeys",
                    "MinimapAttacks","MinimapRightClicks","NumberOfPACs",
                    "GapBetweenPACs","ActionLatency",
                    "ActionsInPAC","TotalMapExplored","WorkersMade","UniqueUnitsMade",
                    "ComplexUnitsMade","ComplexAbilityUsed","MaxTimeStamp"]
df = pd.read_csv("starcraft.csv")
df = df[relatedColumnsList]
df.dropna()
df = df[df['LeagueIndex']!=8]
df.head(n=5)

leagueOne=df[df['LeagueIndex']==1]['Age']
leagueTwo=df[df['LeagueIndex']==2]['Age']
leagueThree=df[df['LeagueIndex']==3]['Age']
leagueFour=df[df['LeagueIndex']==4]['Age']
leagueFive=df[df['LeagueIndex']==5]['Age']
leagueSix=df[df['LeagueIndex']==6]['Age']
leagueSeven=df[df['LeagueIndex']==7]['Age']

leagues=[leagueOne,leagueTwo,leagueThree,leagueFour,
         leagueFive,leagueSix,leagueSeven]
newLabels=["Bronze", "Silver", "Gold", "Platinum",
           "Diamond", "Master", "Grandmaster"]

fig=plt.figure(figsize=(25,15))
plt.title("Player League Number - Age Distribution")
for i in range(len(leagues)):
    leagues[i].hist(alpha=0.9,bins=60,label=newLabels[i])
    plt.legend(loc="best")

leagueOne=df[df['LeagueIndex']==1]['APM']
leagueTwo=df[df['LeagueIndex']==2]['APM']
leagueThree=df[df['LeagueIndex']==3]['APM']
leagueFour=df[df['LeagueIndex']==4]['APM']
leagueFive=df[df['LeagueIndex']==5]['APM']
leagueSix=df[df['LeagueIndex']==6]['APM']
leagueSeven=df[df['LeagueIndex']==7]['APM']

leagues=[leagueOne,leagueTwo,leagueThree,leagueFour,
         leagueFive,leagueSix,leagueSeven]

fig=plt.figure(figsize=(25,15))
plt.title("Player League Number - Action Per Minute Distribution")
for i in range(len(leagues)):
    leagues[i].hist(alpha=0.8,bins=80,label=newLabels[i])
    plt.legend(loc="best")

plt.style.use(['seaborn-dark'])
fig, axes = plt.subplots(nrows=1, ncols = 1, figsize = (15,8))
fig.suptitle('Attribute Relationships', fontsize=25, fontweight='bold')
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
r_matrix = df.corr().round(decimals=1)
sns.heatmap(r_matrix, mask=mask, annot=True, fmt='g',
            annot_kws={'size':10},linewidths=.3,cmap='coolwarm')
plt.show()

willBeFocusedColumns = ['APM','SelectByHotkeys', 'AssignToHotkeys',
                        'NumberOfPACs','GapBetweenPACs', 'ActionLatency']

ySelected = df['LeagueIndex']
xSelected = df[willBeFocusedColumns]

from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest=train_test_split(xSelected,ySelected,test_size=0.33)

from sklearn.tree import DecisionTreeClassifier
dtc= DecisionTreeClassifier()
dtc.fit(xTrain,yTrain)
yPrediction=dtc.predict(xTest)
print("Decision Tree Confusion Matrix")
cm=confusion_matrix(yTest,yPrediction)
print(cm)
print("Score of Decision Tree: ",dtc.score(xTest,yTest),"\n")

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(xTrain,yTrain)
yPrediction=gnb.predict(xTest)
print("Naive Bayes Confusion Matrix")
cm=confusion_matrix(yTest,yPrediction)
print(cm)
print("Score of Naive Bayes: ",gnb.score(xTest,yTest),"\n")

from sklearn import svm
svmTrial = svm.LinearSVC()
svmTrial = svmTrial.fit(xTrain,yTrain)
yPrediction = svmTrial.predict(xTest)
print("Support Vector Machine Confusion Matrix")
cm=confusion_matrix(yTest,yPrediction)
print(cm)
print("Score of Support Vector Machine: ",svmTrial.score(xTest,yTest),"\n")

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(xTrain,yTrain)
yPrediction=rfc.predict(xTest)
print("Random Forest Confusion Matrix")
cm=confusion_matrix(yTest,yPrediction)
print(cm)
print("Score of Random Forest: ",rfc.score(xTest,yTest),"\n")
###############################################################################
######################## ONE VERSUS ALL CLASSIFICATION ########################
###############################################################################


def CalculateOneVsAll(targetRank):
    oneVsAllDataFrame = df.copy(deep=True)
    leagueIndexes=[1,2,3,4,5,6,7]
    if targetRank in leagueIndexes:
        leagueIndexes[targetRank-1]=0
    for i in range(len(oneVsAllDataFrame.index)):  
        if(oneVsAllDataFrame['LeagueIndex'][i]!=targetRank):
            for k in range(len(leagueIndexes)):
                oneVsAllDataFrame['LeagueIndex'].replace(leagueIndexes[k],0,inplace=True)
    
    OVAWillBeFocusedColumns = ['APM','SelectByHotkeys', 'AssignToHotkeys',
                            'NumberOfPACs','GapBetweenPACs', 'ActionLatency']
    yOVASelected = oneVsAllDataFrame['LeagueIndex']
    xOVASelected = oneVsAllDataFrame[OVAWillBeFocusedColumns]        
            
    xOVATrain,xOVATest,yOVATrain,yOVATest=train_test_split(xOVASelected,
                                                           yOVASelected,
                                                           test_size=0.33)  
    ######################DecisionTreeClassifier One vs All####################
    dtcOVA=DecisionTreeClassifier()
    dtcOVA.fit(xOVATrain,yOVATrain)
    yOVAPrediction=dtcOVA.predict(xOVATest)
    print("One Versus All Decision Tree Confusion Matrix")
    cmOVA=confusion_matrix(yOVATest,yOVAPrediction)
    print(cmOVA)
    print(yOVAPrediction)
    print("OVA Score of Decision Tree: ",dtcOVA.score(xOVATest,yOVATest),"\n")
    
    ######################Naive Bayes Classifier One vs All####################
    gnbOVA=GaussianNB()
    gnbOVA.fit(xOVATrain,yOVATrain)
    yOVAPrediction=gnbOVA.predict(xOVATest)
    print("One Versus All Naive Bayes Confusion Matrix")
    cmOVA=confusion_matrix(yOVATest,yOVAPrediction)
    print(cmOVA)
    print(yOVAPrediction)
    print("Score of Naive Bayes: ",gnbOVA.score(xOVATest,yOVATest),"\n")
    ######################Support Vector Machine One vs All####################
    svmOVA = svm.LinearSVC()
    svmOVA = svmOVA.fit(xOVATrain,yOVATrain)
    yOVAPrediction = svmOVA.predict(xOVATest)
    print("One Versus All Support Vector Machine Confusion Matrix")
    cmOVA=confusion_matrix(yOVATest,yOVAPrediction)
    print(cmOVA)
    print(yOVAPrediction)
    print("OVA Score of Support Vector Machine: ",svmOVA.score(xOVATest,yOVATest),
          "\n")
    #########################RandomForestClassifier One vs All#################
    rfcOVA=RandomForestClassifier()
    rfcOVA.fit(xOVATrain,yOVATrain)
    yOVAPrediction=rfcOVA.predict(xOVATest)
    print("One Versus All Random Forest Confusion Matrix")
    cmOVA=confusion_matrix(yOVATest,yOVAPrediction)
    print(cmOVA)
    print(yOVAPrediction)
    print("OVA Score of Random Forest: ",rfcOVA.score(xOVATest,yOVATest),"\n")
    
notifier=["-----ONE VERSUS ALL FOR BRONZE LEAGUE PREDICTIONS-----\n",
          "-----ONE VERSUS ALL FOR SILVER LEAGUE PREDICTIONS-----\n",
          "-----ONE VERSUS ALL FOR GOLD LEAGUE PREDICTIONS-----\n",
          "-----ONE VERSUS ALL FOR PLATINIUM LEAGUE PREDICTIONS-----\n",
          "-----ONE VERSUS ALL FOR DIAMOND LEAGUE PREDICTIONS-----\n",
          "-----ONE VERSUS ALL FOR MASTER LEAGUE PREDICTIONS-----\n",
          "-----ONE VERSUS ALL FOR GRAND MASTER LEAGUE PREDICTIONS-----\n"]
endNotifier=["++++ End of Bronze league predictions++++\n",
          "++++ End of Silver league predictions++++\n",
          "++++ End of Gold league predictions++++\n",
          "++++ End of Platinium league predictions++++\n",
          "++++ End of Diamond league predictions++++\n",
          "++++ End of Master league predictions++++\n",
          "++++ End of Grand Master league predictions++++\n"]
j=1        
while j<=7:
    print(notifier[j-1])
    CalculateOneVsAll(j)
    print(endNotifier[j-1])
    j+=1
    


    
    