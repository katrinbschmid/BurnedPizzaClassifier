"""
"https://github.com/udacity/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%208%20-%20Transfer%20Learning%20(Solution).ipynb
#https://forums.fast.ai/t/using-pytorch-for-kaggle-dogs-vs-cats-competition-tutorial/30233
based on classifies as burned as n
https://discuss.pytorch.org/t/easiest-way-to-draw-training-validation-loss/13195/9
"""
import time
import os
from collections import OrderedDict
 
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn,optim
import torch.nn.functional as F
import torchvision

data_dir = r"D:\workspace\git_scr\BurnedPizzaClassifier\data\pizza"#'Cat_Dog_data'
fp = r'C:\Users\kabis\.pytorch\pizza_pyt.pth'

# TODO: Define transforms for the training data and testing data
train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(255),
        torchvision.transforms.CenterCrop(224),
        torchvision. transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = torchvision.datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = torchvision.datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
val_data = torchvision.datasets.ImageFolder(data_dir + '/validation', transform=train_transforms)

batchs = 32
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batchs, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batchs)
valloader = torch.utils.data.DataLoader(val_data, batch_size=batchs, shuffle=False)
class_names = train_data.classes

def getModel():
    model = torchvision.models.densenet121(pretrained=True)
    #We can load in a model such as DenseNet. Let's print out the model architecture so we can see what's going on.
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(1024, 500)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(500, 2)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
        
    model.classifier = classifier

    for device in ['cpu', 'cuda']:
        criterion = nn.NLLLoss()
        """
        torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
        The negative log likelihood loss. It is useful to train a classification problem with C classes
        """
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
        model.to(device)
        for ii, (inputs, labels) in enumerate(trainloader):
            # Move input and label tensors to the GPU
            inputs, labels = inputs.to(device), labels.to(device)
    
            start = time.time()
            #labels tensor([1, 0, 2,
            #print(inputs, "p inputs, labels", labels)
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if ii==3:
                break
        print(f"p Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")

    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.densenet121(pretrained=True)
    
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
          epochs=1, steps=0, print_every=5):
    running_loss = 0
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
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
                        
                print(f"pEpoch {epoch+1}/{epochs}.. " + str(steps)+
                      f"Train loss: {running_loss/print_every:.3f}.. "
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

#https://raw.githubusercontent.com/pytorch/examples/master/mnist/main.py
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            print( data, target, target.view_as(pred))

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def visualize_model(model, dataloader, device, num_images=6,images_so_far = 0):
    """
    visualize how our model is doing on validation set.
    The above code will show you 6 (by default) images from validation set and show what our model think they are. The predictions should look quite correct at this point.
    """
    misclassified = []
    was_training = model.training
    model.eval()
    fig = plt.figure()    
    with torch.no_grad():#temporarily set all the requires_grad flag to false.
        """
        with mpl.rc_context(rc={'interactive': False}):
    plt.show()
        will make all the operations in the block have no gradients.
    In pytorch, you can't do inplacement changing of w1 and w2, which are two variables with require_grad = True. I think that avoiding the inplacement changing of w1 and w2 is because it will cause error in back propagation calculation. Since inplacement change will totally change w1 and w2.
    However, if you use this no_grad(), you can control the new w1 and new w2 have no gradients since they are generated by operations, which means you only change the value of w1 and w2, not gradient part, they still have previous defined variable gradient information and back propagation can continue
        """
        for i, (inputs, labels) in enumerate(dataloader):
            #print(dir(dataloader.dataset.imgs), "i", i, inputs.size()[0])
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) 
            print(inputs)
            print(i, "outputs", inputs.size()[0], "pred", len(preds),images_so_far)
            for j in range(inputs.size()[0]):
                images_so_far += 1 # starts at 1
                print(j, len(inputs), (num_images//2, 2, images_so_far,dataloader.dataset.imgs[j]))
                if images_so_far > num_images:
                    model.train(mode=was_training)
                    break
                imgn = os.path.basename(dataloader.dataset.imgs[images_so_far -1 + j][0])
                if preds[j] != dataloader.dataset.imgs[images_so_far-1  + j][1]:
                    misclassified.append(imgn)
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}({})'.format(class_names[preds[j]], imgn))
                imshow(inputs.cpu().data[j])
            fig.savefig(r'D:/foop'+str(i)+'.png')
            break
        #model.train(mode=was_training)
        #misclassified.extend(visualize_model(model, dataloader,device, num_images=num_images,images_so_far=images_so_far)[0])
        plt.ioff()#https://stackoverflow.com/questions/12358312/keep-plotting-window-open-in-matplotlib
        #fig.savefig(r'D:/foop.png')
        #plt.show(block=True)
    return misclassified, images_so_far

def main():
    model, device, optimizer, criterion = getModel()
    inputs, classes = next(iter(trainloader))# Make a grid from batch
    sample_train_images = torchvision.utils.make_grid(inputs)
    #helper.imshow(sample_train_images, title=classes)
    if os.path.isfile(fp):
        state_dict = torch.load(fp)
        print(state_dict.keys())
        model.load_state_dict(state_dict)
    else:
        train(model, trainloader, device, optimizer, criterion,
               epochs=10, steps=0, print_every=5)
        torch.save(model.state_dict(), fp)
    test(None, model, device, valloader)
    return 0
    misclassified, images_so_far = visualize_model(model, valloader, device, num_images=6)
    print (images_so_far, " f: ", len(misclassified), misclassified)
    return 0

print(class_names)
main()
print("done")
