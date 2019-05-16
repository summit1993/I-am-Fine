# -*- coding: UTF-8 -*-
import torch.optim as optim
from FSTRN_model import *
from VSR_metrics import *
from load_data import *
import numpy as np

def FSTRN_train(data_set_info_dict, config_info, results_save_dir, model_save_dir):
    data_loaders = get_SVR_loaders(data_set_info_dict, config_info)
    model = FSTRN_Model(config_info['rfb_num'])
    device = config_info['device']
    model = model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config_info['lr'], weight_decay=config_info['weight_decay'])
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(results_save_dir):
        os.makedirs(results_save_dir)
    log_file_name = os.path.join(results_save_dir, 'FSTRN_results')
    model_process(model, data_loaders, optimizer, config_info, log_file_name, model_save_dir)


def model_process(model, loaders, optimizer, config_info,log_file_name_prefix, model_save_dir):
    device = config_info['device']
    epoch_num = config_info['epoch_num']
    show_iters = config_info['show_iters']
    model_save_epoch = config_info['model_save_epoch']
    results = []
    running_loss = 0.0
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    train_loader = loaders['train']
    for epoch in range(epoch_num):
        model.train()
        for step, data in enumerate(train_loader, 0):
            LR_volums, HR_images, LR_R_image = data
            LR_volums, HR_images, LR_R_image = LR_volums.to(device), HR_images.to(device), LR_R_image.to(device)
            optimizer.zero_grad()
            outputs = model([LR_volums, LR_R_image])
            loss = criterion(outputs, HR_images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if show_iters > 0:
                if step % show_iters == (show_iters - 1):
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, step + 1, running_loss / show_iters))
                    running_loss = 0.0

            if model_save_epoch > 0:
                if (epoch + 1) % model_save_epoch == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, os.path.join(model_save_dir, 'checkpoint_' + str(epoch) + '.tar'))

        with torch.no_grad():
            model.eval()
            print('*' * 10, 'Begin to validation', '*' * 10)
            if 'val' in loaders:
                val_loader = loaders['val']
                count = 0.0
                PSNR = 0.0
                for _, val_data in enumerate(val_loader, 0):
                    LR_volums, HR_images, LR_R_image = val_data
                    LR_volums, LR_R_image = LR_volums.to(device), LR_R_image.to(device)
                    outputs = model([LR_volums, LR_R_image])
                    outputs = outputs.to('cpu').numpy()
                    outputs = np.rint(outputs)
                    outputs[outputs < 0] = 0
                    outputs[outputs > 255] = 255
                    HR_images = HR_images.numpy()
                    for i  in range(HR_images.shape[0]):
                           PSNR += cal_img_PSNR(outputs[i], HR_images[i])
                    count += HR_images.shape[0]
                PSNR = PSNR * 1.0 / count
                print('PSNR:\t', PSNR)
                fw = open('val_result_' + str(epoch) + '.txt', 'w')
                fw.write(str(PSNR))
                fw.close()
            print('*' * 10, 'Finish validation', '*' * 10)

