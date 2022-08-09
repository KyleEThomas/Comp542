
# ## Number 2: 

# In[1]:


from sklearn.datasets import load_breast_cancer
import numpy as np

data = load_breast_cancer()

X3 = data.data[:, [2,3]]
y3 = data.target

print('Class labels:', np.unique(y3))


# In[2]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X3, y3, test_size=0.3, random_state=1, stratify=y3)


# In[3]:


print('Labels count in y:', np.bincount(y3))
print('Labels count in y_train:', np.bincount(y_train))
print('Labels count in y_test:', np.bincount(y_test))


# In[4]:


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# To check recent matplotlib compatibility
import matplotlib
from distutils.version import LooseVersion


def plot_decision_regions(X3, y3, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y3))])

    # plot the decision surface
    x1_min, x1_max = X3[:, 0].min() - 1, X3[:, 0].max() + 1
    x2_min, x2_max = X3[:, 1].min() - 1, X3[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z3 = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z3 = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z3, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y3)):
        plt.scatter(x=X3[y == cl, 0], 
                    y=X3[y == cl, 1],
                    alpha=0.8, 
                    color=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X3[test_idx, :], y3[test_idx]

        
        if LooseVersion(matplotlib.__version__) < LooseVersion('0.3.4'):
            plt.scatter(X_test[:, 0],
                        X_test[:, 1],
                        c='',
                        edgecolor='black',
                        alpha=1.0,
                        linewidth=1,
                        marker='o',
                        s=100, 
                        label='test set')
        else:
            plt.scatter(X_test[:, 0],
                        X_test[:, 1],
                        c='none',
                        edgecolor='black',
                        alpha=1.0,
                        linewidth=1,
                        marker='o',
                        s=100, 
                        label='test set')        


# In[5]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[6]:


from sklearn.linear_model import Perceptron

ppn = Perceptron(eta0=0.3, random_state=1)
ppn.fit(X_train_std, y_train)


# In[7]:


from sklearn.metrics import accuracy_score
y_pred = ppn.predict(X_test_std)

print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))


# In[8]:


from sklearn.ensemble import RandomForestClassifier


# In[9]:


# 569 total samples

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

i = 56
forest = RandomForestClassifier(max_samples = i,
                                criterion='gini',
                                n_estimators=25, 
                                random_state=1,
                                n_jobs=2)
forest.fit(X_combined, y_combined)



# In[ ]:


plot_decision_regions(X_combined, y_combined, classifier=forest)

plt.xlabel('X')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_22.png', dpi=300)
plt.show()


# In[ ]:


k = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
i = 1
for i in k:
    forest = RandomForestClassifier(max_features = i)
    forest.fit(X_train, y_train)

    plot_decision_regions(X3, y3, classifier=forest)

    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.tight_layout()
    #plt.savefig('images/03_22.png', dpi=300)
    plt.show()
    i += 1


# In[ ]:




