#https://github.com/udacity/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%208%20-%20Transfer%20Learning%20(Solution).ipynb
#https://forums.fast.ai/t/using-pytorch-for-kaggle-dogs-vs-cats-competition-tutorial/30233
import time
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision

import helper
data_dir = r"C:\Users\kabis\.pytorch\cat\root"#'Cat_Dog_data'
fp = r'C:\Users\kabis\.pytorch\cat\cats_U.pth'

"""
https://github.com/wontheone1/cats-dogs-pytorch
transforms for the training data and testing data
Most of the pretrained models require the input to be 224x224 images. 
Also, we'll need to match the normalization used when the models were trained. 
Each color channel was normalized separately, the means are [0.485, 0.456, 0.406] and the standard deviations are [0.229, 0.224, 0.225]
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

mini batch 4
"""
# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

val_data = datasets.ImageFolder(data_dir + '/validation', transform=train_transforms)
valloader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
class_names = train_data.classes

def getModel():
    model = models.densenet121(pretrained=True)
    #We can load in a model such as DenseNet. Let's print out the model architecture so we can see what's going on.
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(1024, 500)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(500, 2)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
        
    model.classifier = classifier

    for device in ['cpu', 'cuda']:
        criterion = nn.NLLLoss()
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
        model.to(device)
        for ii, (inputs, labels) in enumerate(trainloader):
            # Move input and label tensors to the GPU
            inputs, labels = inputs.to(device), labels.to(device)
            start = time.time()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if ii==3:
                break
            
        print(f"c Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")

    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(256, 2),
                                     nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    model.to(device)
    return model, device, optimizer, criterion

def train(model, trainloader, device, optimizer, criterion,
          epochs=1, steps=0, print_every=10):
    running_loss = 0
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            #print(inputs, "c inputs, labels", labels)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            #Define a Loss function and optimizer
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                """
                model.eval() will set all layers in your model to evaluation mode. This affects layers like dropout layers that turn "off" nodes during training with some probability, but should allow every node to be "on" for evaluation.
                will make all the operations in the block have no gradients.
                In pytorch, you can't do inplacement changing of w1 and w2, which are two variables with require_grad = True. I think that avoiding the inplacement changing of w1 and w2 is because it will cause error in back propagation calculation. Since inplacement change will totally change w1 and w2.
                However, if you use this no_grad(), you can control the new w1 and new w2 have no gradients since they are generated by operations, which means you only change the value of w1 and w2, not gradient part, they still have previous defined variable gradient information and back propagation can continue
                """
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. " + str(steps)+
                      f" Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated# Get a batch of training data

def visualize_model(model, dataloader, device, num_images=6):
    """
    visualize how our model is doing on validation set.
    The above code will show you 6 (by default) images from validation set and show what our model think they are. The predictions should look quite correct at this point.
    """
    import matplotlib.pyplot as plt

    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()    
    with torch.no_grad():#temporarily set all the requires_grad flag to false.
        """
        will make all the operations in the block have no gradients.
        In pytorch, you can't do inplacement changing of w1 and w2, which are two variables with require_grad = True. I think that avoiding the inplacement changing of w1 and w2 is because it will cause error in back propagation calculation. Since inplacement change will totally change w1 and w2.
        However, if you use this no_grad(), you can control the new w1 and new w2 have no gradients since they are generated by operations, which means you only change the value of w1 and w2, not gradient part, they still have previous defined variable gradient information and back propagation can continue
        """
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)    
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) 
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        visualize_model(model_conv)
        #plt.ioff()#https://stackoverflow.com/questions/12358312/keep-plotting-window-open-in-matplotlib
        fig.savefig(r'D:/foocat.png')
        #plt.show(block=True)
        #    plt.close(fig)    # close the figure window

def main():
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.plot([0,1,2], [10,20,3])
    fig.savefig('D:/to.png')   # save the figure to file
    plt.close(fig)    # close the figure window
    
    model, device, optimizer, criterion = getModel()
    inputs, classes = next(iter(trainloader))# Make a grid from batch
    sample_train_images = torchvision.utils.make_grid(inputs)
    helper.imshow(sample_train_images, title=classes)
    train = True
    if train:
        if os.path.isfile(fp):
            state_dict = torch.load(fp)
            print(state_dict.keys())
            model.load_state_dict(state_dict)
        else:
            train(model, trainloader, device, optimizer, criterion)
            torch.save(model.state_dict(), fp)
    visualize_model(model, valloader, device, num_images=6)
print("c")
main()
print("done")
