import numpy as np
from utils import commonutils as utils
import knn
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image, cmap=cm.gray)
    ax.set_title(people.target_names[target])

# count how often each target appears
counts = np.bincount(people.target)
# print counts next to target names
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='   ')
    if (i + 1) % 3 == 0:
        print()

mask = np.zeros(people.target.shape, dtype=np.bool)
print(mask[:5])
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

print(mask.shape)
X_people = people.data[mask]
y_people = people.target[mask]

# scale the grayscale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability
X_people = X_people / 255.


# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)

#KNN
x_train_stan = utils.standardize(X_train)

#standardize using train data
x_t_mean =np.mean(X_train, axis=0)
x_t_std = np.std(X_train, axis=0, ddof=1)
#use the mean and std from train to standardize test
x_test_stan = utils.standardize_m_s(X_test,x_t_mean,x_t_std)

print(x_train_stan.shape)
cov_x = utils.get_covariance(x_train_stan)
e,v = np.linalg.eig(cov_x)

print(e.shape)
print(v.shape)

idx = e.argsort()[::-1]
eig_val = e[idx]
eig_vec = v[:,idx]

print(eig_vec.shape)
eig_100 = eig_vec[:,:100]
eig_val_100 = eig_val[:100]
print(eig_100.shape)

x_train_100 = utils.get_projection(x_train_stan,eig_100)
x_test_100 = utils.get_projection(x_test_stan,eig_100)

#KNN_100
y_pred = knn.KNN(x_train_100,y_train, x_test_100,y_test,1)

#Metrics
accuracy = utils.get_accuracy(y_test,y_pred)
print(accuracy)