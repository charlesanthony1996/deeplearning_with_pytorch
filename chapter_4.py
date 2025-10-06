import torch


# listing 4.1 
import imageio

img_arr = imageio.imread('/users/charles/desktop/images/bobby.jpg')
# print(img_arr.shape)


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
# print(batch.shape)

n_channels = batch.shape[1]
for c in range(n_channels):
    mean = torch.mean(batch[:, c])
    std = torch.std(batch[:, c])
    batch[:, c] = (batch[:, c] - mean) / std


import imageio

dir_path = '/users/charles/desktop/images/data/p1ch4/volumetric-dicom/2-LUNG_3.0_B70f-04083'
vol_arr = imageio.volread(dir_path, 'DICOM')
# print(vol_arr.shape)

vol = torch.from_numpy(vol_arr).float()
vol = torch.unsqueeze(vol, 0)

# print(vol.shape)

# listing 4.3

import csv
import numpy as np

wine_path = '/users/charles/desktop/images/data/p1ch4/tabular-wine/winequality-white.csv'
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=';', skiprows=1)
# print(wineq_numpy)

col_list = next(csv.reader(open(wine_path), delimiter=';'))
# print(wineq_numpy.shape, col_list)

wineq = torch.from_numpy(wineq_numpy)

print(wineq.shape, wineq.dtype)

# representing scores
data = wineq[:, :-1]
print(data, data.shape)

target = wineq[:, :-1]
print(target, target.shape)

target = wineq[:, -1].long()
print(target)

target_onehot = torch.zeros(target.shape[0], 10)

target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

target_unsqueezed = target.unsqueeze(1)
print(target_unsqueezed)

data_mean = torch.mean(data, dim = 0)
print(data_mean)

data_var = torch.var(data, dim = 0)
print(data_var)

data_normalized = (data - data_mean) / torch.sqrt(data_var)
print(data_normalized)

bad_indexes = target <= 3
print(bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum())

bad_data = data[bad_indexes]
print(bad_data.shape)

bad_data = data[target <= 3]
mid_data = data[(target > 3) & (target < 7)]
good_data = data[target >= 7]


bad_mean = torch.mean(bad_data, dim = 0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim = 0)

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))

total_sulfur_threshold = 141.83
total_sulphur_data = data[:, 6]
predicted_indexes = torch.lt(total_sulphur_data, total_sulfur_threshold)

print(predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum())

actual_indexes = target > 5

print(actual_indexes.shape, actual_indexes.dtype, actual_indexes.sum())

n_matches = torch.sum(actual_indexes & actual_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()

print(n_matches, n_matches / n_predicted, n_matches / n_actual)




# listing 4.4

bikes_numpy = np.loadtxt("/users/charles/desktop/dlwpt-code/data/p1ch4/bike-sharing-dataset/hour-fixed.csv", 
                         dtype=np.float32, delimiter=",", skiprows=1, converters={1:lambda x:float(x[8:10])})
# print(bikes_numpy)
bikes = torch.from_numpy(bikes_numpy)
# print(bikes)

print(bikes.shape, bikes.stride())

daily_bikes = bikes.view(-1, 24, bikes.shape[1])
print(daily_bikes.shape, daily_bikes.stride())

