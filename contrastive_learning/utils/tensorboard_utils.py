"""


Created on: Saturday, April 25th, 2021
Author: Jacob A Rose

"""





# Add a "projector" to TensorBoard
## We can visualize the lower dimensional representation of higher dimensional data via the add_embedding method
### Source: https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

# helper function
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# # select random images and their target indices
# images, labels = select_n_random(trainset.data, trainset.targets)
# # get the class labels for each image
# class_labels = [classes[lab] for lab in labels]
# # log embeddings
# features = images.view(-1, 28 * 28)
# writer.add_embedding(features,
#                     metadata=class_labels,
#                     label_img=images.unsqueeze(1))
# writer.close()