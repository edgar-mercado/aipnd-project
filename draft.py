%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch
from torch import nn, optim

import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import datasets, transforms, models

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
t_training = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
t_validation = train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
t_testing =  transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
# TODO: Load the datasets with ImageFolder
#image_datasets = datasets.ImageFolder(data_dir, transform=data_transforms)
ds_training = datasets.ImageFolder(train_dir, transform=t_training)
ds_testing = datasets.ImageFolder(test_dir, transform=t_validation)
ds_validation = datasets.ImageFolder(valid_dir, transform=t_testing)

# TODO: Using the image datasets and the trainforms, define the dataloaders
#dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=32, shuffle=True)
dl_training = torch.utils.data.DataLoader(ds_training, batch_size=50, shuffle=True)
dl_testing = torch.utils.data.DataLoader(ds_testing, batch_size=50, shuffle=False)
dl_validation = torch.utils.data.DataLoader(ds_validation, batch_size=50, shuffle=True)

class_to_idx = ds_training.class_to_idx

#############
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

images, labels = next(iter(dl_training))
print("size")
print(len(images[0,2]))

# TODO: Build and train your network
def create_model(hidden_layers, class_to_idx):
    model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier_input_size = model.classifier.in_features
    print("Input size: ", classifier_input_size)
    output_size = 102

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_input_size, hidden_layers)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_layers, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    model.class_to_idx = class_to_idx
    return model

hidden_layers = 512
model = create_model(hidden_layers, class_to_idx)

def validate(model, criterion, data_loader):
    accuracy = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in iter(data_loader):
            if torch.cuda.is_available():
                inputs = Variable(inputs.float().cuda())
                labels = Variable(labels.long().cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)

            output = model.forward(inputs)
            test_loss += criterion(output, labels)

            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

    return test_loss/len(data_loader), accuracy/len(data_loader)

def train(model, epochs, learning_rate, criterion, optimizer, training_loader, validation_loader):

    model.train()
    print_every = 40
    steps = 0

    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in iter(training_loader):
            steps += 1

            if torch.cuda.is_available():
                with torch.no_grad():
                    inputs = Variable(inputs.float().cuda())
                    labels = Variable(labels.long().cuda())
            else:
                with torch.no_grad():
                    inputs = Variable(inputs)
                    labels = Variable(labels)

            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss, accuracy = validate(model, criterion, validation_loader)

                print("Epoch: {}/{} ".format(epoch+1, epochs),
                        "Training Loss: {:.3f} ".format(running_loss/print_every),
                        "Validation Loss: {:.3f} ".format(validation_loss),
                        "Validation Accuracy: {:.3f}".format(accuracy))

epochs = 8
learning_rate = 0.001
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
train(model,
      epochs,
      learning_rate,
      criterion,
      optimizer,
      dl_training,
      dl_validation)
print("Training completed")

# TODO: Do validation on the test set

test_loss, accuracy = validate(model, criterion, dl_testing)

print(f"Validation Accuracy: {accuracy:.3f}.. "
      f"Validation Loss: {test_loss:.3f}")


# TODO: Save the checkpoint
checkpoint_path = 'my_checkpoint.pth'

state = {
    'arch': 'densenet121',
    'learning_rate': learning_rate,
    'hidden_layers': hidden_layers,
    'epochs': epochs,
    'state_dict': model.state_dict(),
    'optimizer' : optimizer.state_dict(),
    'class_to_idx' : model.class_to_idx
}
with torch.no_grad():
    torch.save(state, checkpoint_path)

# TODO: Write a function that loads a checkpoint and rebuilds the model
checkpoint_path = 'my_checkpoint.pth'
state = torch.load(checkpoint_path)
learning_rate = state['learning_rate']
class_to_idx = state['class_to_idx']

model = create_model(hidden_layers, class_to_idx)

model.load_state_dict(state['state_dict'])
optimizer.load_state_dict(state['optimizer'])

print("Loaded '{}' (arch={}, hidden_layers={}, epochs={})".format(
    checkpoint_path,
    state['arch'],
    state['hidden_layers'],
    state['epochs']))

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    size = 224
    # TODO: Process a PIL image for use in a PyTorch model
    width, height = image.size

    if height > width:
        height = int(max(height * size / width, 1))
        width = int(size)
    else:
        width = int(max(width * size / height, 1))
        height = int(size)

    resized_image = image.resize((width, height))

    x0 = (width - size) / 2
    y0 = (height - size) / 2
    x1 = x0 + size
    y1 = y0 + size
    cropped_image = image.crop((x0, y0, x1, y1))
    np_image = np.array(cropped_image) / 255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image_array = (np_image - mean) / std
    np_image_array = np_image.transpose((2, 0, 1))

    return np_image_array

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
        model = model.cuda()
    else:
        model = model.cpu()
    image = Image.open(image_path)
    np_array = process_image(image)
    tensor = torch.from_numpy(np_array)
    if use_gpu:
        with torch.no_grad():
            var_inputs = Variable(tensor.float().cuda())
    else:
        var_inputs = Variable(tensor, volatile=True)
    var_inputs = var_inputs.unsqueeze(0)
    output = model.forward(var_inputs)
    ps = torch.exp(output).data.topk(topk)
    probabilities = ps[0].cpu() if use_gpu else ps[0]
    classes = ps[1].cpu() if use_gpu else ps[1]
    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    for label in classes.numpy()[0]:
        mapped_classes.append(class_to_idx_inverted[label])
    return probabilities.numpy()[0], mapped_classes

image_path = test_dir + '/58/image_02752.jpg'
probabilities, classes = predict(image_path, model)

print(probabilities)
print(classes)

# TODO: Display an image along with the top 5 classes
image_path = test_dir + '/58/image_02752.jpg'
probabilitiess, classes = predict(image_path, model)
max_index = np.argmax(probabilities)
max_probability = probabilities[max_index]
label = classes[max_index]

fig = plt.figure(figsize=(6,6))
ax1 = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)
ax2 = plt.subplot2grid((15,9), (9,2), colspan=5, rowspan=5)

image = Image.open(image_path)
ax1.axis('off')
ax1.set_title(cat_to_name[label])
ax1.imshow(image)
labels = []
for cl in classes:
    labels.append(cat_to_name[cl])
y_pos = np.arange(5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels)
ax2.invert_yaxis()

ax2.set_xlabel('Probability')
ax2.barh(y_pos, probabilities, xerr=0, align='center')

plt.show()
