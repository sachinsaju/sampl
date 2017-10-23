from sklearn.tree import DecisionTreeClassifier
#data
features=[[140,0],[130,0],[150,1],[170,1]]
#soft=0 1= bumpy
labels=[0,0,1,1]
#0=apple 1=orange


#data classifier
clf=DecisionTreeClassifier()
clf.fit(features,labels)

#prediction
p=clf.predict([[160,1]])
print("predicion = ",p)

