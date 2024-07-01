import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms, utils
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import math

def corrupt_input(x):
    corrupting_matrix = 2.0 * torch.rand_like(x)

    return x * corrupting_matrix

class Encoder(nn.Module):
  def __init__(self, n_in, n_hidden_1, n_hidden_2, n_hidden_3, n_out):
    super(Encoder, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_in, n_hidden_1, bias=True),
        nn.BatchNorm1d(n_hidden_1),
        nn.Sigmoid())
    self.layer2 = nn.Sequential(
        nn.Linear(n_hidden_1, n_hidden_2, bias=True),
        nn.BatchNorm1d(n_hidden_2),
        nn.Sigmoid())
    self.layer3 = nn.Sequential(
        nn.Linear(n_hidden_2, n_hidden_3, bias=True),
        nn.BatchNorm1d(n_hidden_3),
        nn.Sigmoid())
    self.layer4 = nn.Sequential(
        nn.Linear(n_hidden_3, n_out, bias=True),
        nn.BatchNorm1d(n_out),
        nn.Sigmoid())
    
  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    return self.layer4(x)

class Decoder(nn.Module):
  def __init__(self, n_in, n_hidden_1, n_hidden_2, n_hidden_3, n_out):
    super(Decoder, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Linear(n_in, n_hidden_1, bias=True),
        nn.BatchNorm1d(n_hidden_1),
        nn.Sigmoid())
    self.layer2 = nn.Sequential(
        nn.Linear(n_hidden_1, n_hidden_2, bias=True),
        nn.BatchNorm1d(n_hidden_2),
        nn.Sigmoid())
    self.layer3 = nn.Sequential(
        nn.Linear(n_hidden_2, n_hidden_3, bias=True),
        nn.BatchNorm1d(n_hidden_3),
        nn.Sigmoid())
    n_size = math.floor(math.sqrt(n_out))
    self.layer4 = nn.Sequential(
        nn.Linear(n_hidden_3, n_out, bias=True),
        nn.BatchNorm1d(n_out),
        nn.Sigmoid(),
        nn.Unflatten(1, torch.Size([1, n_size,n_size])))
    
  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    #print('Shape of decoder output: ', x.shape)
    return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(corrupt_input)
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

encoder = Encoder(784,1000,500,250,2).to(device)
decoder = Decoder(2,250,500,1000,784).to(device)



loss_fn = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr = 0.001, 
                       betas=(0.9,0.999), 
                       eps=1e-08)


trainset = datasets.MNIST('.',
                          train=True, 
                          transform=transform,
                          download=True)
trainloader = DataLoader(trainset,
                         batch_size=32,
                         shuffle=True)

writer = SummaryWriter('./corrupt_autoencoder=2_logs')

# Training Loop 
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    for input, labels in trainloader:
        input = input.to(device)
        optimizer.zero_grad()
        code = encoder(input)
        output = decoder(code)
        #print(input.shape, output.shape)
        loss = loss_fn(output, input)
        loss.backward()
        optimizer.step()

    writer.add_scalar('Loss/train', loss.item(), epoch)
    print(f"Epoch: {epoch} Loss: {loss}")

writer.close()

i = 0
encoder.eval()
decoder.eval()
with torch.no_grad():
  for images, labels in trainloader:
    print(images.shape)
    images = images.to(device)
    if i == 3:
      break
    grid = utils.make_grid(images).cpu()
    plt.figure()
    plt.imshow(grid.permute(1,2,0))
    
    code = encoder(images)
    output = decoder(code)
    
    grid = utils.make_grid(output).cpu()
    plt.figure()
    plt.imshow(grid.permute(1,2,0))
    plt.show()
    i += 1



### There will be a folder named "mnist_autoencoder=2_logs" created in ./
### Then run "tensorboard --logdir .\mnist_autoencoder=2_logs\" in powershell
### It will make you to go to http://localhost:6006/
### You can see Tensorboard outputs there