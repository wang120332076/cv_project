import io
import requests
import numpy
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable

LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
IMG_URL = 'https://s3.amazonaws.com/outcome-blog/wp-content/uploads/2017/02/25192225/cat.jpg'

# define normalization process
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Scale(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])
# download image & label dictionary
response = requests.get(IMG_URL)
img_pil = Image.open(io.BytesIO(response.content))
labels = {int(key):value for (key, value)
          in requests.get(LABELS_URL).json().items()}
# get normalized
img_tensor = preprocess(img_pil)
img_tensor.unsqueeze_(0)

# use pre-trained VGG16
vgg16 = models.vgg16_bn(pretrained = True)
img_variable = Variable(img_tensor)
fc_out = vgg16(img_variable)

print(labels[fc_out.data.numpy().argmax()])

################### VGG-16
# create pre-trained VGG16
# vgg16 = models.vgg16(pretrained = True)
# for param in vgg16.parameters():
#     param.requires_grad = False     # freeze parameters

# # modify FC layers
# vgg16.classifier._modules['6'] = nn.Linear(4096, 172)
# if use_gpu:
#     vgg16 = vgg16.cuda()

# # prepare for training
# criterion = nn.CrossEntropyLoss()

# # Observe that only parameters of final layer are being optimized as
# # opoosed to before.
# optimizer_conv = optim.SGD(vgg16.classifier._modules['6'].parameters(),
#                            lr=0.001, momentum=0.9)

# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# # start training
# vgg16 = train_model(vgg16, criterion, optimizer_conv, exp_lr_scheduler, 25)

# # show results
# visualize_model(vgg16)

# # save trained model
# torch.save(vgg16, './vgg16_out.pth')

################### ResNet-18
# resnet = models.resnet18(True)
# for param in resnet.parameters():
#     param.requires_grad = False     # freeze parameters

# resnet.fc = nn.Linear(512, 172)
# # resnet.fc = nn.Linear(2048, 172)  # for ResNet-101
# if use_gpu:
#     resnet = resnet.cuda()
# criterion = nn.CrossEntropyLoss()
# optimizer_conv = optim.SGD(resnet.fc.parameters(),
#                            lr=0.001, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
# resnet = train_model(resnet, criterion, optimizer_conv, exp_lr_scheduler, 25)

# visualize_model(resnet)

# # save trained model
# torch.save(resnet, './resnet18_out.pth')

# example code for visualizing images in dataset
# im, lab_n = next(iter(dataloaders['train']))
# out = torchvision.utils.make_grid(im)
# imshow(out, [labels[x] for x in lab_n])