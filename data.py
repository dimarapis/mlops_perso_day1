import torch
import numpy as np


import cv2


def mnist():
    # exchange with the corrupted mnist dataset
    
    train = []
    for i in range(5):
        with np.load(f'../../../data/corruptmnist/train_{i}.npz') as data:
            
            train.append([torch.Tensor(data['images']), torch.from_numpy(data['labels'])])
            
    with np.load(f'../../../data/corruptmnist/test.npz') as data:
        test = [torch.Tensor(data['images']), torch.from_numpy(data['labels'])]
    
    return train, test
'''




def asd(path):
    with np.load(path, allow_pickle=True) as f:  # pylint:
        #disable=unexpected-keyword-arg
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)

path = "../../../data/corruptmnist/test_.npz"
dataset = np.load(path)
print(dataset.files)
print(len(dataset['labels']))
print((len(dataset['images'])))
image_sample = 0
title = 'Label is {label}'.format(label=dataset['labels'][image_sample])

np_image =  dataset['images'][image_sample]
print(np_image.shape)
resized_image = cv2.resize(np_image, (200, 200)) 

cv2.imshow(title, resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#(train_images, train_labels), (test_images, test_labels ) = asd(path=path)

#print(test_labels)

'''

def vis_corruption():
        
    print("visualize corruption")
    _, test_set = mnist()
    images,labels = test_set
    for sample_image in range(10):
        
        title = 'Label is {label}'.format(label=labels[sample_image])

        np_image =  images[sample_image].squeeze().numpy()#dataset['images'][image_sample]
        print(np_image.shape)
        resized_image = cv2.resize(np_image, (200, 200)) 

        cv2.imshow(title, resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()