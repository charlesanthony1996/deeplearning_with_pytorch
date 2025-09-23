a = [1.0, 2.0, 3.0]

# print(a[0])

a[2] = 3.0

# print(a)

import torch
a = torch.ones(3)
# print(a)

# print(a[1])

# print(float(a[1]))

a[2] = 2.0
# print(a[2])

points = torch.zeros(6)
points[0] = 4.0
points[1] = 1.0
points[2] = 5.0
points[3] = 3.0
points[4] = 2.0
points[5] = 1.0

points = torch.tensor([4.0, 1.0, 5.0, 3.0, 2.0, 1.0])
# print(points)


# print(float(points[0]), float(points[1]))

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# print(points)


# print(points.shape)

torch.Size([3, 2])

points = torch.zeros(3, 2)
# print(points)

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# print(points)

# print(points[0, 1])

# print(points[0])

some_list = list(range(6))
some_list[:]
some_list[1:4]
some_list[1:]
some_list[:4]
some_list[:-1]
some_list[1:4:2]

points[1:]
points[1:, :]
points[1:, 0]
points[None]

img_t = torch.randn(3, 5, 5)
weights = torch.tensor([0.2126, 0.7152, 0.07222])

batch_t = torch.randn(2, 3, 5, 5)

img_gray_naive = img_t.mean(-3)
batch_gray_naive = batch_t.mean(-3)
img_gray_naive.shape, batch_gray_naive.shape


print(img_gray_naive.shape)
print(batch_gray_naive.shape)

unsqueeze_weights = weights.unsqueeze(-1).unsqueeze_(-1)
img_weights = (img_t * unsqueeze_weights)
batch_weights = (batch_t * unsqueeze_weights)
img_gray_weighted = img_weights.sum(-3)
batch_gray_weighted = batch_weights.sum(-3)

# print(batch_weights)
# print(batch_t.shape)
# print(unsqueeze_weights.shape)

img_gray_weighted_fancy = torch.einsum('...chw, c-> ...hw', img_t, weights)
batch_gray_weighted_fancy = torch.einsum('...chw, c-> ...hw', batch_t, weights)
print(batch_gray_weighted_fancy.shape)

