import numpy as np
import torch
import torch.nn as nn


class NormDenorm:
    # Store mean and std for transforms, apply normalization and de-normalization
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def norm(self, img, tensor=False):
        # normalize image to feed to network
        if tensor:
            return (img - float(self.mean[0])) / float(self.std[0])
        else:
            return (img - self.mean) / self.std

    def denorm(self, img, cpu=True, variable=True):
        # reverse normalization for viewing
        if cpu:
            img = img.cpu()
        if variable:
            img = img.data
        img = img.numpy().transpose(1, 2, 0)
        return img * self.std + self.mean


class TensorTransform(nn.Module):
    # Used to convert between default color space and VGG colorspace
    def __init__(self, res=256, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(TensorTransform, self).__init__()

        self.mean = torch.zeros([3, res, res]).cuda()
        self.mean[0, :, :] = mean[0]
        self.mean[1, :, :] = mean[1]
        self.mean[2, :, :] = mean[2]

        self.std = torch.zeros([3, res, res]).cuda()
        self.std[0, :, :] = std[0]
        self.std[1, :, :] = std[1]
        self.std[2, :, :] = std[2]

    def forward(self, x):
        norm_ready = (x * 0.5) + 0.5
        result = (norm_ready - self.mean) / self.std
        return result


def test_single(denorm, tensor_norm, dataloader, style, model, save=False):
    # Show and save
    transform = load.NormDenorm([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tensor_transform = n.TensorTransform(
        res=params["res"], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    image = cv2_open(style_path)
    resize = ResizeCV(size)
    image = resize({"image": image})["image"]
    image = torch.FloatTensor(np.rollaxis(transform.norm(image), 2))

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5 * 1))
    model.eval()
    real = dataloader.next()
    real_vgg = Variable(real[0].cuda())
    real_def = Variable(real[1].cuda())
    test = tensor_norm(model(real_def))
    ax[0, 0].cla()
    ax[0, 0].imshow(denorm.denorm(real_vgg[0]))
    ax[0, 1].cla()
    ax[0, 1].imshow(denorm.denorm(test[0]))
    ax[0, 2].cla()
    ax[0, 2].imshow(denorm.denorm(style[0]))
    model.train()
    if save:
        plt.savefig(save)
    else:
        plt.show()
    plt.close(fig)
