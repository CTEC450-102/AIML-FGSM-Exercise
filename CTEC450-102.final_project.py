"""
Raynard Forte II
CTEC 450-102
Final Project:  AI Model Attack and Defense
April 30, 2026
Professor Carter

This code is written in Python.  It includes a program that builds an AI model using the MNIST handwritten digits dataset.  This data set is used to test whether a system can read handwritten numbers and guess, based on the pattern of the colors in the image analyzed in the data set, the number based on a pattern (Neural Nine, 2023).

Additionally, code will then be added to the process that is adversarial.  This process will Ftion to decrease the accuracy of the AI model's prediction of the numbers in the output of the original program.

Finally, another process will be included to mitigate and/or reverse the attack and improve the accuracy of the output by a significant amount.  
"""

#import statements to input datasets and statistical tools to operate on data.
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt




trainData = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

testData = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


#trainData.data
"""

Used to output data and labels.

print(f'MNIST training data: {trainData.data}\n')
print(f'MNIST training data labels: {trainData.targets}\n')
print(f'MNIST test data: {testData.data}\n')
print(f'MNIST test data labels: {testData.targets}\n')
"""

#use dataloaders to load the data in batches for training and testing.
loaders = {
    "train": DataLoader(trainData, batch_size=64, shuffle=True, num_workers=1),
    "test": DataLoader(testData, batch_size=64, shuffle=True, num_workers=1),
}

#loaders used for testing in the main function for the attack phase.
testLoader = DataLoader(datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),])), batch_size=1, shuffle=True)

trainLoader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),])), batch_size=1, shuffle=True)

#define model architecture for neural network model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        #build convolution layers here
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mdl = CNN().to(device)
optimizer = optim.Adam(mdl.parameters(), lr=0.001)

loss_fn = nn.CrossEntropyLoss()


#uncomment the block for basic training and testing of the model below without robustness feature added to the process.
#placing dataset in training mode and running the training loop for 10 epochs.  This will be used to train the model on the data and improve accuracy of the output.
#this will also send the data to the device for processing.  Setting the gradients to zero, running the forward pass, calculating the loss, running the backward loss, and updating the weights of the model are all included in this block of python code.
def train(epoch):
    mdl.train()
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = mdl(data)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()
        
        if batch_idx % 20 == 0:
            
             print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders["train"].dataset)} ({100. * batch_idx / len(loaders["train"]):.0f}%)]\tLoss: {loss.item():.6f}')

def test():
    mdl.eval()
    trainLoss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in loaders['train']:
            data, target = data.to(device), target.to(device)
            output = mdl(data)
            trainLoss += loss_fn(output, target).item()
            predictions = output.argmax(dim=1, keepdim=True)
            correct += predictions.eq(target.view_as(predictions)).sum().item()
    trainLoss /= len(loaders['train'].dataset)
    print(f'\nTrain set: \nAverage loss: {trainLoss:.4f}')
    print(f'Accuracy: {correct}/{len(loaders["train"].dataset)} ({100. * correct / len(loaders["train"].dataset):.0f}%)\n')
    print("="*50)



"""
Attack Phase:

The implementation of the Fast Gradient Sign Method (FGSM) is written in the block below.  It's function is gradually distort the accuracy of the output of the code. This will demonstrate the value of securing AI models from adversarial inputs.
Variable 'epsi' is an array variable containing epsilon values for use in subsequent functions.

(MLWorks, 2026)
"""

epsi = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
torch.manual_seed(42)

def fgsm_attack(image, epsi, data_grad):
    #set sign of the data gradient	equal to the parameter data_grad.sign()
    sign_data_grad = data_grad.sign()
    
    #Make a sum of the image in the dataset and the epsilon value times the sign of the data gradient in the Fast Gradient Sign Method being used (Goodfellow, Shlens, and Szegedy, 2014).
    perturbedImage = image + epsi * sign_data_grad
    
    #clamp the range of the perturbed image to values between 0 and 1 to gradually distort the accuracy of the outputs.
    perturbedImage = torch.clamp(perturbedImage, 0, 1)
    return perturbedImage


def denorm(batch, mean=[0.1307], std=[0.3081]):
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def trainAttack(mdl, device, trainLoader, epsi):
    correct = 0
    adv_examples = []
    for data, target in trainLoader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = mdl(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue
        
        loss = F.nll_loss(output, target)

        #set gradients back to zero.
        mdl.zero_grad()
        loss.backward() #calculate gradients of model in backward pass.
        data_grad = data.grad.data #collect the data gradient.
        data_denorm = denorm(data) #denormalize the data.
        perturbedData = fgsm_attack(data_denorm, epsi, data_grad) #call FGSM to create adversarial example.
        perturbedDataNormalized = transforms.Normalize((0.1307,), (0.3081,))(perturbedData) #normalize the perturbed data.
        
        output = mdl(perturbedDataNormalized) #Construct output for model of normalized perturbed data.
        final_pred = output.max(1, keepdim=True)[1] #provide the index value.
        
        if final_pred.item() == target.item():
            correct += 1
            if epsi == 0 and len(adv_examples) < 5: #particular examples for instances when epsilon value is 0.
                adv_ex = perturbedData.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
                
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbedData.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
                
    finAcc = correct / float(len(trainLoader)) #deteremine the accurany of the model under attack after all calculations are performed.
    print(f'Epsilon: {epsi} \tTrain Accuracy: {correct} / {len(trainLoader)} = {finAcc*100:.2f}%')
    return finAcc, adv_examples

"""
Defense Phase:
Here we retrain the model using adversarial training noise.  This should make the model more resistant to attack.
"""
"""
def train(epoch, epsilon=0.2):
    mdl.train()
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)

        data.requires_grad = True
        output = mdl(data)
        loss = loss_fn(output, target)
    
        loss.backward()
        optimizer.zero_grad()

        #constructing adversarial examples to be used for training.
        data_grad = data.grad.detach().clone()
        adv_data = fgsm_attack(data, epsilon, data_grad)

        #introduce the adversarial examples into the training process by calculating the loss on both the original and adversarial data, and then backpropagating the combined loss to update the model's weights.
        output_adv = mdl(adv_data)
        loss_adv = loss_fn(output_adv, target)

        #Calculate the total loss.
        total_loss = (loss + loss_adv)/2

        #optimize the model and updating the weights.
        optimizer.zero_grad()
        loss_adv.backward()
        optimizer.step()

        
        if batch_idx % 20 == 0:
            print(f'Retrain Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders["train"].dataset)} ({100. * batch_idx / len(loaders["train"]):.0f}%)]\tLoss: {total_loss.item():.6f}')
"""


if __name__ == "__main__":
    
    print("Training Phase: \n ")
    print("="*50)
    """
    for epoch in range(1, 11):
        train(epoch)
        test()
    
    #place the model in evaluation mode for testing.  Then sample is displayed to show image with the prediction of the corresponding number.
    
    mdl.eval()
    data, target = trainData[0]
    data = data.unsqueeze(0).to(device)
    output = mdl(data)
    prediction = output.argmax(dim=1, keepdim=True)
    print(f'Prediction = {prediction}')
    image = data.squeeze(0).squeeze(0).cpu().numpy()
    plt.imshow(image, cmap='gray')
    plt.title('Data Sample')
    plt.show()
    (MLWorks, 2026)
    """
    
    #Defense Phase written in the block below.  
    for epoch in range(1, 11):
        train(epoch)
        test()

                      

    #attack phase is displayed below.  The accuracy of the results decreases with each iteration of the loop.
    print("Attack Phase: \n")
    print("="*50)
    accRate = [] #variable array to contain the rates of accuracy from the calculations of the attack phase.
    examples = [] #variable array to contain the examples of the attack phase.

    for eps in epsi:
        acc, ex = trainAttack(mdl, device, trainLoader, eps)
        accRate.append(acc)
        examples.append(ex)

    # The plt function is called to provide visualization of the data over time as the epsilon values increase.
    plt.figure(figsize=(5,5))
    plt.plot(epsi, accRate, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, 0.35, step=0.05))
    plt.title('Proportion of Accurate Predictions with Increasing Epsilon Values')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy Rate')
    plt.show()

    counter = 0
    plt.figure(figsize=(8,10))

    for i in range(len(epsi)):
        for j in range(len(examples[i])):
            counter += 1
            plt.subplot(len(epsi), len(examples[0]), counter)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f"Eps: {epsi[i]}", fontsize=12)
            orig, adv, ex = examples[i][j]
            plt.title(f'{orig} -> {adv}')
            plt.imshow(ex, cmap='gray')
    plt.tight_layout()
    plt.show()

    print("="*50)
    
"""
References:

MLWorks (2026, March 31).  'Adversarial Attacks In Machine Learning: Full Tutorial with Code.'  YouTube.  Date Accessed:  May 1, 2026.  Site Accessed: https://www.youtube.com/watch?v=jbTy7aZ9cZ8

Neural Nine (2023, August 22).  'PyTorch Project: Handwritten Digit Recognition' YouTube.  Date Accessed:  April 30, 2026.  Site Accessed:  https://www.youtube.com/watch?v=vBlO87ZAiiw



"""