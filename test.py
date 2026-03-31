import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from torch.utils.data import DataLoader
from collections import OrderedDict

from utils import AverageMeter, write_img, chw_to_hwc, pad_img
from datasets.loader import PairLoader
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gunet_t', type=str)
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--data_dir', default='./data/', type=str)
parser.add_argument('--save_dir', default='./saved_models/', type=str)
parser.add_argument('--result_dir', default='./results/', type=str)
parser.add_argument('--test_set', default='SOTS-IN', type=str)
parser.add_argument('--exp', default='reside-in', type=str)
args = parser.parse_args()


# ================= SSIM =================
def calculate_ssim(img1, img2):

    def _ssim_single_channel(x, y, data_range=255):
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        x = x.astype(np.float64)
        y = y.astype(np.float64)

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu_x = cv2.filter2D(x, -1, window)[5:-5, 5:-5]
        mu_y = cv2.filter2D(y, -1, window)[5:-5, 5:-5]

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = cv2.filter2D(x * x, -1, window)[5:-5, 5:-5] - mu_x_sq
        sigma_y_sq = cv2.filter2D(y * y, -1, window)[5:-5, 5:-5] - mu_y_sq
        sigma_xy = cv2.filter2D(x * y, -1, window)[5:-5, 5:-5] - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

        return ssim_map.mean()

    if img1.ndim == 2:
        return _ssim_single_channel(img1, img2)
    elif img1.ndim == 3:
        return np.mean([
            _ssim_single_channel(img1[..., i], img2[..., i])
            for i in range(3)
        ])
    else:
        raise ValueError("Invalid image shape")


# ================= Load Model =================
def single(save_dir):
    state_dict = torch.load(save_dir)['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v

    return new_state_dict


# ================= Test Function =================
def test(test_loader, network, result_dir):

    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()
    network.eval()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

    for idx, batch in enumerate(test_loader):

        input = batch['source'].cuda()
        target = batch['target'].cuda()
        filename = batch['filename'][0]

        with torch.no_grad():

            H, W = input.shape[2:]
            input = pad_img(input, network.patch_size if hasattr(network, 'patch_size') else 16)

            output = network(input).clamp_(-1, 1)
            output = output[:, :, :H, :W]

            # [-1,1] → [0,1]
            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

        # ================= Convert to numpy =================
        output_np = output.permute(0, 2, 3, 1).cpu().numpy()
        target_np = target.permute(0, 2, 3, 1).cpu().numpy()

        for out_img, tgt_img in zip(output_np, target_np):

            out_img = (out_img * 255.0).astype(np.uint8)
            tgt_img = (tgt_img * 255.0).astype(np.uint8)

            # ===== PSNR =====
            psnr_val = cv2.PSNR(out_img, tgt_img)

            # ===== SSIM =====
            ssim_val = calculate_ssim(out_img, tgt_img)

            PSNR.update(psnr_val)
            SSIM.update(ssim_val)

        print(f'Test: [{idx}]\tPSNR: {PSNR.val:.02f} ({PSNR.avg:.02f})\tSSIM: {SSIM.val:.03f} ({SSIM.avg:.03f})')

        f_result.write('%s,%.02f,%.03f\n' % (filename, psnr_val, ssim_val))

        out_img_save = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, 'imgs', filename), out_img_save)

    f_result.close()

    os.rename(
        os.path.join(result_dir, 'results.csv'),
        os.path.join(result_dir, '%.03f | %.04f.csv' % (PSNR.avg, SSIM.avg))
    )


# ================= Main =================
def main():

    network = eval(args.model)()
    network.cuda()

    saved_model_dir = os.path.join(args.save_dir, args.exp, args.model + '.pth')

    if os.path.exists(saved_model_dir):
        print('==> Start testing, model:', args.model)
        network.load_state_dict(single(saved_model_dir))
    else:
        print('==> No trained model found!')
        exit(0)

    dataset_dir = os.path.join(args.data_dir, args.test_set)
    test_dataset = PairLoader(dataset_dir, 'test')

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True
    )

    result_dir = os.path.join(args.result_dir, args.test_set, args.exp, args.model)

    test(test_loader, network, result_dir)


if __name__ == '__main__':
    main()