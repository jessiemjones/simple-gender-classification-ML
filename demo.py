from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

#height, weight, and shoe size
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43],[171,58,41]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male', 'female']

#sample to be tested
sample = [[157,47,38]]

#decision tree
clf1 = tree.DecisionTreeClassifier()
clfDT = clf1.fit(X,Y)
predictionDT = clfDT.predict(sample)
probabilityDT = clfDT.predict_proba(sample)

#support vector machine
clf2 = svm.SVC(probability=True)
clfSVC =clf2.fit(X,Y)
predicitonSVC = clfSVC.predict(sample)
probabilitySVC = clfSVC.predict_proba(sample)

#kneighbors classifier for 3 args
clf3 = KNeighborsClassifier(n_neighbors=3)
clfN = clf3.fit(X,Y)
predictionN = clfN.predict(sample)
probabilityN = clfN.predict_proba(sample)

#gaussian
clf4 = GaussianProcessClassifier()
clfG = clf4.fit(X,Y)
predictionG = clfG.predict(sample)
probabilityG = clfG.predict_proba(sample)


print('predictionDT classified sample data {} to be {} with a probability of {}'.format(sample, predictionDT, probabilityDT))
print('predicitonSVC classified sample data {} to be {} with a probability of {}'.format(sample, predicitonSVC, probabilitySVC))
print('predictionN classified sample data {} to be {} with a probability of {}'.format(sample, predictionN, probabilityN))
print('predictionG classified sample data {} to be {} with a probability of {}'.format(sample, predictionG, probabilityG))
