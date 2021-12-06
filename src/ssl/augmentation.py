from __future__ import division
import torch
import math
import random
import torch.nn as nn

import torchvision.transforms
from PIL import Image
import matplotlib.pyplot as plt

try:
    import accimage
except ImportError:
    accimage = None
import warnings
import numpy as np

from torchvision.transforms import functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pset):
        coord = None
        for t in self.transforms:
            if 'RandomResizedCropCoord' in t.__class__.__name__:
                img, qset, coord = t(img, pset)
            elif 'FlipCoord' in t.__class__.__name__:
                img, qset, coord = t(img, qset, coord)
            else:
                img = t(img)
            # assert qset
        return img, qset, coord

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomHorizontalFlipCoord(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, pset, coord):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            coord_new = coord.clone()
            coord_new[0] = coord[2]
            coord_new[2] = coord[0]
            w, h = img.size
            qset = [(w - pos[0], pos[1]) if pos else None for pos in pset]
            return F.hflip(img), qset, coord_new
        return img, pset, coord

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlipCoord(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, pset, coord):
        """
        Args:
            img (PIL Image): Image to be flipped.
            pset : [(w1, h1), ... (wn, hn)]
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            coord_new = coord.clone()
            coord_new[1] = coord[3]
            coord_new[3] = coord[1]
            w, h = img.size
            qset = [(pos[0], h - pos[1]) if pos else None for pos in pset]
            return F.vflip(img), qset, coord_new
        return img, pset, coord

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCropCoord(object):
    """Crop the given PIL Image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped, control image area size
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped, control shape
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w, height, width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w, height, width

    def __call__(self, img, pset):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
            pset: Position set, [(w1, h1), ..., (wn, hn)]
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w, height, width = self.get_params(img, self.scale, self.ratio)
        coord = torch.Tensor([float(j) / (width - 1), float(i) / (height - 1),
                              float(j + w - 1) / (width - 1), float(i + h - 1) / (height - 1)])
        qset = [
            (pos[0] - j, pos[1] - i) if (pos[0] >= j and pos[0] <= j + w and pos[1] >= i and pos[1] <= i + h) else None
            for pos in pset]
        qset = [(pos[0] * self.size[0] / w, pos[1] * self.size[0] / h) if pos else None for pos in qset]
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), qset, coord

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class GaussianBlurCoord(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size, sigma):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()
        self.sigma = np.random.uniform(sigma[0], sigma[1])

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * self.sigma * self.sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


def augmentation(im, pset, crop_lb=0.5, crop_ub=1.0, crop_scale_lb=3. / 4., crop_scale_ub=4. / 3., v_f=0.05, h_f=0.25,
                  cj=(0.4, 0.4, 0.4, 0.1), cj_p=0.8, gs=0.2, gb=3, gb_sigma=(0.1, 2.0)):
    """
    pset: node position set, qset is the transformed position set. [(w1, h1), (w2, h2), ..., (wn, hn)]
    scale: hyper-parameter to control returned area / original area.
    size: returned image size.
    returned-qset: node position set. [(w1, h1), None, ..., (wn, hn)]. When returned image doesn't include the pixel,
                     correspondingly return None.
    """
    # Utilize
    img_transform = Compose([
        RandomResizedCropCoord(size=256, scale=(crop_lb, crop_ub), ratio=(crop_scale_lb, crop_scale_ub)),
        RandomVerticalFlipCoord(p=v_f),
        RandomHorizontalFlipCoord(p=h_f),
        transforms.RandomApply([transforms.ColorJitter(cj[0], cj[1], cj[2], cj[3])], p=cj_p),
        transforms.RandomGrayscale(p=gs),
        GaussianBlurCoord(gb, sigma=gb_sigma),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    # img_transform = Compose([
    #     RandomResizedCropCoord(size=256, scale=(0.5, 0.8), ratio=(3. / 4., 4. / 3.)),
    #     RandomVerticalFlipCoord(p=0.05),
    #     RandomHorizontalFlipCoord(p=0.25),
    #     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    #     transforms.RandomGrayscale(p=0.2),
    #     # transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    #     # transforms.RandomSolarize(0.1, p=0.5)
    #     # transforms.ToTensor(),
    #     # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #     #                      std=[0.229, 0.224, 0.225])
    # ])
    trans, qset, coord = img_transform(im, pset=pset)
    return trans, qset, coord


if __name__ == '__main__':
    im = Image.open("./Cars_000a.png").convert('RGB')
    # pset = [(101, 101), (453, 256), (400, 300), (567, 432), (621, 571)]
    pset = [(40, 62), (129, 32), (98, 104), (225, 231), (234, 102)]
    color_list = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
    plt.imshow(im)
    width, height = im.size
    for i in range(len(pset)):
        p = pset[i]
        if p is not None:
            plt.scatter((p[0],), (p[1],), c=color_list[i % len(color_list)])
    plt.savefig("./car_test_img-scatter.jpg")
    plt.close()

    trans, qset, coord = augmentation(im, pset)

    plt.imshow(trans)
    for i in range(len(qset)):
        q = qset[i]
        if q is not None:
            plt.scatter((q[0],), (q[1],), c=color_list[i % len(color_list)])
    plt.savefig("./car_test_img-transform.jpg")
    transform1 = transforms.Compose([
        transforms.ToTensor(),
    ])
    trans_tensor = transform1(trans)
    print("Hello World~")
