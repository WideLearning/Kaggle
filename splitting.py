import easyocr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch import nn
from matplotlib.cm import get_cmap


def rgb_array():
    plt.draw()
    fig = plt.gcf()
    fig.canvas.draw()
    result = np.array(fig.canvas.renderer.buffer_rgba())
    plt.clf()
    return result[:, :, :3].astype(np.uint8)


def quantize(arr, s=100, k=4):
    arr = torch.tensor(arr, dtype=float) / 255
    assert 0 <= arr.min() and arr.max() <= 1
    pool = nn.AdaptiveAvgPool2d(s)
    arr = pool(arr.permute(2, 0, 1)).permute(1, 2, 0)
    print(arr.shape)
    arr = (k * arr).int()
    _clusters, inverse = torch.unique(arr.view(-1, 3), dim=0, return_inverse=True)
    return inverse.reshape(arr.shape[:-1]).numpy()


def matching(a, b, removal_cost=0.5):
    n, m = len(a), len(b)
    dist = np.linalg.norm(a.reshape(n, 1, 2) - b.reshape(1, m, 2), axis=-1)
    assert dist.shape == (n, m)
    row_ind, col_ind = linear_sum_assignment(dist)
    cost = dist[row_ind, col_ind].sum()
    # for i, j in zip(row_ind, col_ind):
    #     ay, ax = a[i]
    #     by, bx = b[j]
    #     plt.plot([ax, bx], [ay, by], lw=1)
    # plt.show()
    # add all these lines together
    return cost + abs(n - m) * removal_cost


def EMD(x, y):
    def where(pic, col):
        return np.argwhere(pic == col) / pic.shape

    return sum(
        matching(where(x, c), where(y, c)) for c in range(max(x.max(), y.max()) + 1)
    )


x = np.arange(10)
y = np.random.randn(*x.shape)
plt.plot(x, y)
plt.title(
    "The quick brown fox jumps over the lazy dog",
    fontfamily="serif",
    fontsize=14,
    fontstyle="italic",
)
first = rgb_array()
first = quantize(first)

y += 5 * np.random.randn(*x.shape)
plt.plot(x, y)
plt.title(
    "The quick brown fox jumps over the lazy dog",
    fontfamily="serif",
    fontsize=15,
    fontstyle="italic",
)
second = rgb_array()
# second = quantize(second)

# plt.imshow(second)
# plt.show()

# print(EMD(first, second))

vertical = second.std(axis=1).mean(axis=-1)
horizontal = second.std(axis=0).mean(axis=-1)
plt.plot(horizontal)
plt.show()
# zero segments

# reader = easyocr.Reader(["en"]) 
# result = reader.readtext(second, min_size=1, text_threshold=0.1, low_text=0.1, link_threshold=0.1)
# cmap = get_cmap("rainbow")
# plt.imshow(second)
# for rect, text, score in result:
#     rect = np.array(rect)
#     # rect[:, 1] = second.shape[1] - rect[:, 1]
#     plt.plot(*zip(*rect, rect[0]), c=cmap(score))
#     x = rect[:, 0].min() * 0.9 + rect[:, 0].max() * 0.1
#     y = rect[:, 1].min() * 0.5 + rect[:, 1].max() * 0.5
#     # plt.text(s=f"{text}", x=x, y=y, fontsize=10)
# plt.show()
# print(result)
