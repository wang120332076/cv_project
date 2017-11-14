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