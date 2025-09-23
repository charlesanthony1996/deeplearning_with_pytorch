import torch


# listing 4.1 
import imageio

img_arr = imageio.imread('/users/charles/desktop/images/bobby.jpg')
print(img_arr.shape)


img = torch.from_numpy(img_arr)
out = img.permute(2, 0, 1)
# print(out)

batch_size = 3
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)



import os
data_dir = '/users/charles/desktop/images/image-cats/'
filenames = [name for name in os.listdir(data_dir) if os.path.splitext(name)[-1] == '.png']

for i, filename in enumerate(filenames):
    img_arr = imageio.imread(os.path.join(data_dir, filename))
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2, 0, 1)
    img_t = img_t[:3]
    batch[i] = img_t


batch = batch.float()
print(batch.shape)

n_channels = batch.shape[1]
for c in range(n_channels):
    mean = torch.mean(batch[:, c])
    std = torch.std(batch[:, c])
    batch[:, c] = (batch[:, c] - mean) / std


import imageio

dir_path = '/users/charles/desktop/images/data/p1ch4/volumetric-dicom/2-LUNG_3.0_B70f-04083'
vol_arr = imageio.volread(dir_path, 'DICOM')
print(vol_arr.shape)

vol = torch.from_numpy(vol_arr).float()
vol = torch.unsqueeze(vol, 0)

print(vol.shape)

# listing 4.3

import csv
import numpy as np

wine_path = '/users/charles/desktop/images/data/p1ch4/tabular-wine/winequality-white.csv'
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=';', skiprows=1)
# print(wineq_numpy)

col_list = next(csv.reader(open(wine_path), delimiter=';'))
print(wineq_numpy.shape, col_list)

