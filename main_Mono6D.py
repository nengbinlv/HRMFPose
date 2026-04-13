import os
import torch
import numpy as np
from models.networkMono6D import ContourPose
from dataset.DatasetMono6D import MyDataset
from torch.utils.data import DataLoader
from evalMono6D import evaluator
# import eval.evaluator as evaluator
import argparse
import time
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import scipy.io
# writer = SummaryWriter(log_dir='./runs/' + 'temp')
import datetime
import random

writer = SummaryWriter('./runs' + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

cuda = torch.cuda.is_available()

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def train(model, dataloader, optimizer, device, antoWeightedModel, epoch):
    model.train()
    total_loss = 0.0
    heatmap_loss_sum = 0.0
    contour_loss_sum = 0.0
    vis_loss_num = 0.0
    grah_loss_num = 0.0
    mask_loss_num = 0.0
    iter = 0
    start = time.time()

    for data in dataloader:
        iter += 1
        if cuda:

            img, heatmap, K, pose, gt_contour, gt_kpVis, gt_grah, gt_mask, gt_mask_dilate = [x.to(device) for x in data]
        else:
            img, heatmap, K, pose, gt_contour, gt_kpVis, gt_grah, gt_mask, gt_mask_dilate= data

        loss = model(img, heatmap, gt_contour, gt_kpVis, gt_grah, gt_mask, gt_mask_dilate, epoch)

        final_loss = (torch.mean(loss["heatmap_loss"]) + torch.mean(loss["contour_loss"]) + torch.mean(loss["vis_loss"]) +
                    torch.mean(loss["grah_loss"]) + torch.mean(loss["mask_loss"]))

        final_loss = final_loss.to(torch.float32)

        heatmap_loss = torch.mean(loss["heatmap_loss"]).item()
        vis_loss = torch.mean(loss["vis_loss"]).item()
        grah_loss = torch.mean(loss["grah_loss"]).item()
        contour_loss = torch.mean(loss["contour_loss"]).item()
        mask_loss = torch.mean(loss["mask_loss"]).item()

        heatmap_loss_sum += heatmap_loss
        contour_loss_sum += contour_loss
        mask_loss_num += mask_loss
        vis_loss_num += vis_loss
        grah_loss_num += grah_loss
        loss_item = final_loss.item()
        total_loss += loss_item

        # final_loss_auto = antoWeightedModel(heatmap_loss, contour_loss, vis_loss, grah_loss, mask_loss)
        # final_loss_auto = final_loss_auto.to(torch.float32)

        if iter % 10 == 0:
            print(f'loss:{loss_item:.6f}  heatmap_loss:{heatmap_loss:.6f}  contour_loss:{contour_loss:.6f}  vis_loss:{vis_loss:.6f} '
                  f' grah_loss:{grah_loss:.6f}  mask_loss:{mask_loss:.6f}  '  )  #

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
    duration = time.time() - start
    print('Time cost:{}'.format(duration))

    return (total_loss / len(dataloader.dataset), heatmap_loss_sum / len(dataloader.dataset),\
           contour_loss_sum / len(dataloader.dataset), vis_loss_num / len(dataloader.dataset),
            grah_loss_num / len(dataloader.dataset), mask_loss_num / len(dataloader.dataset))


def load_network(net, model_dir, optimizer, resume=True, epoch=-1, strict=False):
    if not resume:
        return 0
    if not os.path.exists(model_dir):
        return 0
    pths = [int(pth.split(".")[0]) for pth in os.listdir(model_dir) if "pkl" in pth]
    if len(pths) == 0:
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch

    print("Load model: {}".format(os.path.join(model_dir, "{}.pkl".format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, "{}.pkl".format(pth)))
    try:
        net.load_state_dict(pretrained_model['net'], strict=strict)
        optimizer.load_state_dict(pretrained_model['optimizer'])
    except KeyError:
        net.load_state_dict(pretrained_model, strict=strict)
    return pth


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.5 ** (epoch // 20))
    print("LR:{}".format(lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


@torch.no_grad()
def get_wd_params(model: nn.Module):
    # Parameters must have a defined order.
    # No sets or dictionary iterations.
    # See https://pytorch.org/docs/stable/optim.html#base-class
    # Parameters for weight decay.
    all_params = tuple(model.parameters())
    wd_params = list()
    for m in model.modules():
        if isinstance(
                m,
                (
                        nn.Linear,
                        nn.Conv1d,
                        nn.Conv2d,
                        nn.Conv3d,
                        nn.ConvTranspose1d,
                        nn.ConvTranspose2d,
                        nn.ConvTranspose3d,
                ),
        ):
            wd_params.append(m.weight)
    # Only weights of specific layers should undergo weight decay.
    no_wd_params = []
    for p in all_params:
        if p.dim() == 1:
            no_wd_params.append(p)
    assert len(wd_params) + len(no_wd_params) == len(all_params), "Sanity check failed."
    return wd_params, no_wd_params


def main_test(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("using {} device".format(device))

    if args.train:
        train_set = MyDataset(args.data_path, args.class_type, is_train=True)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    if args.eval:
        test_set = MyDataset(args.data_path, args.class_type, is_train=False, scene=args.scene, index=args.index)
        test_loader = DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

    corners = np.loadtxt(
        os.path.join(os.getcwd(), "keypoints/{}.txt".format(args.class_type)))  # KEYPOINTS

    ContourNet = ContourPose(heatmap_dim=corners.shape[0] + 1, graph_dim=(corners.shape[0] * (corners.shape[0]-1)//2) * 2)

    # ContourNet = nn.DataParallel(ContourNet, device_ids=[0])

    ContourNet = ContourNet.to(device)
    wd_params, no_wd_params = get_wd_params(ContourNet)

    # no_wd_params = no_wd_params + list(awl.parameters())
    # {'params': list(awl.parameters()),'lr': 0.0001, 'weight_decay': 0}
    optimizer = torch.optim.AdamW([{'params': list(no_wd_params), 'weight_decay': 0},
                                   {'params': list(wd_params)}
                                   ],
                                  lr=args.lr, weight_decay=0.1)
    # print(list(no_wd_params))
    # print( list(wd_params))
    # print(list(awl.parameters()))

    model_path = os.path.join(os.getcwd(), "weights", args.class_type)

    if args.train:

        start_epoch = load_network(ContourNet, model_path, optimizer) + 1

        for epoch in range(start_epoch, args.epochs + 1):
            print("current class:{}".format(args.class_type))
            loss, heatmap_loss_sum, contour_loss_sum, vis_loss_num, grah_loss_num, mask_loss_num = train(ContourNet, train_loader, optimizer, device, epoch)
            adjust_learning_rate(optimizer, epoch, args.lr)
            print(f'Epoch: {epoch:02d}, Loss: {loss * args.batch_size:.4f}')
            if epoch % 10 == 0:
                if not os.path.exists(os.path.join(os.getcwd(), 'model', args.class_type)):
                    os.makedirs(os.path.join(os.getcwd(), 'model', args.class_type))
                state = {'net': ContourNet.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, os.path.join('model', args.class_type, '{}.pkl'.format(epoch)))
            # writer.add_scalar('train_loss', loss, epoch)
            writer.add_scalars('epoch_metric', {'loss': loss }, epoch)
            writer.add_scalars('epoch_metric', {'heatmap_loss_sum': heatmap_loss_sum }, epoch)
            writer.add_scalars('epoch_metric', {'contour_loss_sum': contour_loss_sum }, epoch)
            writer.add_scalars('epoch_metric', {'vis_loss_num': vis_loss_num}, epoch)
            writer.add_scalars('epoch_metric', {'grah_loss_num': grah_loss_num}, epoch)
            writer.add_scalars('epoch_metric', {'mask_loss_num': mask_loss_num}, epoch)
    if args.eval:
        ContourNet_eval = evaluator(args, ContourNet, test_loader, device)
        load_network(ContourNet, model_path, optimizer, epoch=args.used_epoch)
        ContourNet_eval.evaluate()
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=bool, default=True)
    parser.add_argument("--data_path", type=str, default=os.path.join(os.getcwd(), "data"))
    parser.add_argument("--class_type", type=str, default="Bracket")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--gpu_id", help="GPU_ID", type=str, default="0")
    parser.add_argument("--used_epoch", type=int, default=150)

    args = parser.parse_args()
    main_test(args)
