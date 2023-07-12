# Torch Model Compression(tomoco)

This is a Deep Learning Pruning Package. This package allows you to prune layers of Convolution Layers based on L1 or L2 Norm.
[Tomoco Package](https://pypi.org/project/tomoco/)

## Package install:

```python

pip install tomoco

```


## Usage:

```python

pruner(model= model,experiment_name = "test" , config = config , input_dims = input_dims , pruning_stratergy= pruning_stratergy, pruning_percent= pruning_percent,  train_loader= train_loader, valid_loader= valid_loader)

* model - Any Model with Convolutional Layers are supported
* experiment_name - Provide an experiment name to keep track of the model changes
* config - Make a config class as described below:

    class config:
        lr = 0.001 
        n_classes = 10			 # Intended for output classes
        epochs = 5                         # Set no. of training epochs
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    # Pick the device availble
        batch_size = 64			# Set batch size
        optim = 0
        training =1                        # Set training to 1 if you would like to train post to prune
        criterion = nn.CrossEntropyLoss()  # Set your criterion here
* input_dims - Input dimensions that the model can take in
* pruning_stratergy - Norm (L1, L2) #Only two norms are currently supported; Future scope of dynamic channel selection
* pruning_percent - Percentage of which channels for each Convolutional Block would be removed
* train_loader - Training Data  [ Loaded in a pytorch dataloader ]
* valid_loader - Validation Data [ Loaded in a pytorch dataloader ]




```

## Channel Pruning based on Norm:

```python
from tomoco import pruner
import timm
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader



class config:
    lr = 0.001 
    n_classes = 10			 # Intended for output classes
    epochs = 5                         # Set no. of training epochs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    # Pick the device availble
    batch_size = 64			# Set batch size
    optim = 0
    training =1                        # Set training to 1 if you would like to train post to prune
    criterion = nn.CrossEntropyLoss()  # Set your criterion here

train_dataset = CIFAR10(root='data/', download=True, transform=transforms.ToTensor())
valid_dataset = CIFAR10(root='data/',  download=True,train=False, transform=transforms.ToTensor())

# define the data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size, shuffle=False)


#Use a cutom model or pull a model from a repository

res50 = timm.create_model("resnet50", pretrained=True).to(config.device)
config.optim =  torch.optim.Adam(res50.parameters(), config.lr=0.001,  amsgrad=True) 

pruner(res50,"res50", config, (3,64,64), "L1", 0.15,  train_loader, valid_loader)


```



### To-Do

- [x] Universal Channel-Wise Pruning
- [x] Update Tutorials
- [+] Fine Grained Purning (In-Progress)
- [ ] Quantisation
- [ ] Universal AutoML package
- [ ] Introduction of Sparsification in Pipeline
