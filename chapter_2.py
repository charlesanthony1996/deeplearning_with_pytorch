# https://github.com/deep-learning-with-pytorch/dlwpt-code

from torchvision import models
from torchvision import transforms

# print(dir(models))

alexnet = models.AlexNet()
# resnet = models.resnet101(pretrained=True)
resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)


# print(resnet)

from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

from PIL import Image

img = Image.open("/users/charles/desktop/images/bobby.jpg")
# print(img)

img_t = preprocess(img)

import torch
batch_t = torch.unsqueeze(img_t, 0)
resnet.eval()
# print(resnet.eval())

out = resnet(batch_t)
# print(out)

with open("/users/charles/desktop/images/imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

_, index = torch.max(out, 1)
# _, indices = torch.topk(out, 5)


percentage = torch.nn.functional.softmax(out, dim = 1)[0] * 100
# print(percentage)
# print(labels[index[0]], percentage[index[0]].item())

# for idx in indices[5]:
#     print(labels[idx], percentage[idx].item)

print(labels[index[0]])
print(percentage[index[0]].item())

_, indices = torch.sort(out, descending=True)
print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])


import torch
import torch.nn as nn

class ResNetBlock(nn.Module): # <1>

    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x) # <2>
        return out


class ResNetGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9): # <3> 

        assert(n_blocks >= 0)
        super(ResNetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input): # <3>
        return self.model(input)
    

netG = ResNetGenerator()

model_path = "/users/charles/downloads/horse2zebra_0.4.0.pth"
model_data = torch.load(model_path)
netG.load_state_dict(model_data)

netG.eval()

from PIL import Image
from torchvision import transforms

preprocess = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])

img = Image.open("/users/charles/downloads/horse.jpg")
# img = Image.open("/users/charles/desktop/images/horse2.webp")
# img = Image.open("/users/charles/desktop/images/horse3.jpg")

# img.show()

img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

batch_out = netG(batch_t)

out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_img = transforms.ToPILImage()(out_t)
# out_img.save("/users/charles/desktop/deeplearning-with-pytorch/zebra.jpg")
# out_img.show()



