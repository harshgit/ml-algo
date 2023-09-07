import numpy as np
from utils import commonutils as utils
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import matplotlib.cm as cm



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

#standardize
x_people_stan = utils.standardize(X_people)

#PCA with 100D

cov_x = utils.get_covariance(x_people_stan)
e,v = np.linalg.eig(cov_x)

idx = e.argsort()[::-1]
eig_val = e[idx]
eig_vec = v[:,idx]

eig_100 = eig_vec[:,:100]
eig_val_100 = eig_val[:100]

x_people_100 = utils.get_projection(x_people_stan,eig_100)

def k_means(x,k):
    np.random.seed(0)
    idx = np.random.choice(x.shape[0],k, replace=False)
    x_random_k = x[idx]
    center = np.array([k for k in x_random_k])
    cluster = [[] for k in center]
    i = 0
    while(True):
        cluster = [[] for k in center]
        for data in x:
            sim =  utils.compute_similarity(data,center)
            c_k = np.argmin(sim)
            cluster[c_k].append(data)
            
        new_center = get_centers(cluster)
        d = np.sqrt(utils.compute_similarity(new_center, center))[0]
        if(d < 2**(-23) or i>=10000):
            break
        center = new_center
        i+=1
        
    print(f"Total iterations: {i}")
    return cluster

    
def get_centers(cluster):
    return np.array([np.mean(np.array(cl),axis=0) for cl in cluster])


cluster = k_means(x_people_100,10)