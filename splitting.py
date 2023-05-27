import easyocr
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import get_cmap
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch import nn


def rgb_array():
    plt.draw()
    fig = plt.gcf()
    fig.canvas.draw()
    result = np.array(fig.canvas.renderer.buffer_rgba())
    plt.clf()
    return result[:, :, :3].astype(np.uint8)


def max_sum_segment(arr: np.array, dx: int = 0) -> tuple[int, int]:
    """Returns the segment with the largest sum, shifted by dx.

    Args:
        arr (np.array): Given array.
        dx (int): Index for the first element in arr. Default 0.

    Returns:
        tuple[int, int]: [l, r]
    """
    assert arr.ndim == 1
    n = len(arr)
    l, l_sum = 0, 0  # sum on [0, l - 1]
    lr, lr_sum = (dx, dx), 0
    r_sum = 0  # sum on [0, r]
    for r in range(n):
        r_sum += arr[r]
        if l_sum > r_sum:
            l, l_sum = r + 1, r_sum
        if lr_sum < r_sum - l_sum:
            lr, lr_sum = (l + dx, r + dx), r_sum - l_sum
    return lr


# x = np.arange(10)
# y = np.random.randn(*x.shape)
# plt.plot(x, y)
# plt.title(
#     "The quick brown fox jumps over the lazy dog",
#     fontfamily="serif",
#     fontsize=14,
#     fontstyle="italic",
# )
# arr = rgb_array()
with Image.open("rotated.jpg") as im:
    arr = np.asarray(im)
print(arr.shape)

rectangles = []


def split_rec(ly, ry, lx, rx, min_size=3, min_split=2, splitting_reward=0.2):
    if min(rx - lx, ry - ly) < min_size:
        return
    s = arr[ly:ry, lx:rx].std(axis=1).mean(axis=-1)
    syl, syr = max_sum_segment(splitting_reward * s.mean() - s, dx=ly)
    s = arr[ly:ry, lx:rx].std(axis=0).mean(axis=-1)
    sxl, sxr = max_sum_segment(splitting_reward * s.mean() - s, dx=lx)
    if sxr - sxl > max(syr - syl, min_split):
        split_rec(ly, ry, lx, sxl)
        split_rec(ly, ry, sxr, rx)
    elif syr - syl > min_split:
        split_rec(ly, syl, lx, rx)
        split_rec(syr, ry, lx, rx)
    else:
        rectangles.append((lx, rx, ly, ry))


split_rec(0, arr.shape[0], 0, arr.shape[1])
print(len(rectangles))
for lx, rx, ly, ry in rectangles:
    plt.plot([lx, lx, rx, rx, lx], [ly, ry, ry, ly, ly], c="r")
plt.imshow(arr)
plt.show()
