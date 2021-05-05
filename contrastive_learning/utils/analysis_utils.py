"""

Created on: Saturday, April 25th, 2021
Author: Jacob A Rose

"""



# # (used in the `plot_classes_preds` function below)
# def matplotlib_imshow(img, one_channel=False):
#     if one_channel:
#         img = img.mean(dim=0)
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     if one_channel:
#         plt.imshow(npimg, cmap="Greys")
#     else:
#         plt.imshow(np.transpose(npimg, (1, 2, 0)))


# ######################## 

# ### Source: https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
# def images_to_probs(net, images):
#     '''
#     Generates predictions and corresponding probabilities from a trained
#     network and a list of images
#     '''
#     output = net(images)
#     # convert output probabilities to predicted class
#     _, preds_tensor = torch.max(output, 1)
#     preds = np.squeeze(preds_tensor.numpy())
#     return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


# def plot_classes_preds(net, images, labels):
#     '''
#     Generates matplotlib Figure using a trained network, along with images
#     and labels from a batch, that shows the network's top prediction along
#     with its probability, alongside the actual label, coloring this
#     information based on whether the prediction was correct or not.
#     Uses the "images_to_probs" function.
#     '''
#     preds, probs = images_to_probs(net, images)
#     # plot the images in the batch, along with predicted and true labels
#     fig = plt.figure(figsize=(12, 48))
#     for idx in np.arange(4):
#         ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
#         matplotlib_imshow(images[idx], one_channel=True)
#         ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
#             classes[preds[idx]],
#             probs[idx] * 100.0,
#             classes[labels[idx]]),
#                     color=("green" if preds[idx]==labels[idx].item() else "red"))
#     return fig

# #####################################


# # Source: https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

# # 1. gets the probability predictions in a test_size x num_classes Tensor
# # 2. gets the preds in a test_size Tensor
# # takes ~10 seconds to run
# class_probs = []
# class_label = []
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         output = net(images)
#         class_probs_batch = [F.softmax(el, dim=0) for el in output]

#         class_probs.append(class_probs_batch)
#         class_label.append(labels)

# test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
# test_label = torch.cat(class_label)

# # helper function
# def add_pr_curve_tensorboard(class_index, test_probs, test_label, global_step=0):
#     '''
#     Takes in a "class_index" from 0 to 9 and plots the corresponding
#     precision-recall curve
#     '''
#     tensorboard_truth = test_label == class_index
#     tensorboard_probs = test_probs[:, class_index]

#     writer.add_pr_curve(classes[class_index],
#                         tensorboard_truth,
#                         tensorboard_probs,
#                         global_step=global_step)
#     writer.close()

    
# def add_per_class_pr_curves_tensorboard(test_probs, test_label, global_step=0):
#     # plot all the pr curves
#     for i in range(len(classes)):
#         add_pr_curve_tensorboard(i, test_probs, test_label)
