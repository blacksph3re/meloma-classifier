import pandas as pd
import torchvision
import torch
from PIL import Image
import numpy as np
from ray import tune
import os

class MyDataset(torch.utils.data.Dataset):
  def __init__(self, data_folder, transform, data):
    self.files = [data_folder + '/' + x + '.jpg' for x in data['image_name']]
    self.labels = [0 if x == 'benign' else 1 for x in data['benign_malignant']]
    self.transform = transform
  
  def __len__(self):
    #return len(self.files)
    return 100
  
  def __getitem__(self, idx):
    return (self.transform(Image.open(self.files[idx])), self.labels[idx])
  
  def get_untransformed(self, idx):
    return Image.open(self.files[idx])

def load_data(hparams, device):
  print(os.listdir('.'))

  data_csv = pd.read_csv(hparams['data_csv'])

  train_mask = np.random.rand(len(data_csv)) < 0.9

  data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

  data_csv_train = data_csv[train_mask]
  data_csv_eval = data_csv[~train_mask]

  dataset = MyDataset(hparams['data_folder'], data_transforms, data_csv_train)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=hparams['batch_size'], shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

  dataset_eval = MyDataset(hparams['data_folder'], data_transforms, data_csv_eval)
  dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=128, num_workers=4)

  return dataloader, dataloader_eval

def create_model(hparams, device):
  model = torchvision.models.resnet18(pretrained=True)
  # First go for feature extraction mode
  for param in model.parameters():
    param.requires_grad = False

  # Reinitialize the last layer of the model
  model.fc = torch.nn.Linear(512, 2)

  model.to(device)

  # The optimizer only needs to see the layers which have grads
  params_to_update = []
  print("Trainable parameters:")
  for name,param in model.named_parameters():
      if param.requires_grad:
          print("\t", name)
          params_to_update.append(param)

  optimizer = torch.optim.Adam(params_to_update, lr=hparams['lr'])
  criterion = torch.nn.CrossEntropyLoss()

  return model, optimizer, criterion

def train_epoch(dataloader, model, optimizer, criterion, device):
  model.train()
  total_loss = 0
  epochs = 0
  for images, labels in dataloader:
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.step()
    total_loss += loss.detach().cpu()
    epochs += 1
  return (total_loss / epochs)

def eval(dataloader_eval, model, criterion, device):
  with torch.no_grad():
    total_loss = 0
    epochs = 0
    for images, labels in dataloader_eval:
      images = images.to(device)
      labels = labels.to(device)
      total_loss += criterion(model(images), labels)
      epochs += 1
    return (total_loss / epochs)

class Trainable(tune.Trainable):
  def default_resource_request(self, config):
    return tune.resources.Resources(
      cpu=4,
      gpu=1,
    )

  def _setup(self, hparams):
    self.hparams = hparams
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.data, self.data_eval = load_data(hparams, self.device)
    self.model, self.optimizer, self.criterion = create_model(hparams, self.device)

  def _train(self):
    train_loss = train_epoch(self.data, self.model, self.optimizer, self.criterion, self.device)
    eval_loss = eval(self.data_eval, self.model, self.criterion, self.device)

    return {
      "train_loss": train_loss.item(),
      "eval_loss": eval_loss.item(),
    }
  
  def _save(self, checkpoint_dir):
    toch.save(self.model, os.path.join(checkpoint_dir, 'model.pth'))
  
  def _restore(self, checkpoint_dir):
    self.model = torch.load(os.path.join(checkpoint_dir, 'model.pth'))



analysis = tune.run(
  Trainable,
  name='melanoma_1',
  stop={"training_iteration": 2},
  checkpoint_freq=5,
  config={
    'lr': 1e-4,
    'batch_size': tune.grid_search([8, 64]),
    'checkpoint_dir': os.getcwd() + '/checkpoints',
    'data_csv': os.getcwd() + '/train.csv',
    'data_folder': os.getcwd() + '/data/train',
  }
)

print("Best config: ", analysis.get_best_config(metric="eval_loss", mode='min'))