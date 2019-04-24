import pickle
import torch
import torch.nn as nn
import os
from utilities.my_metrics import Top_K_Right

def model_process(model, loaders, optimizer, config_info, log_file_name_prefix, model_save_dir):
    device = config_info['device']
    epoch_num = config_info['epoch_num']
    show_iters = config_info['show_iters']
    model_save_epoch = config_info['model_save_epoch']
    results = []
    running_loss = 0.0
    train_loader = loaders['train']
    for epoch in range(epoch_num):
        epoch_result = {}
        for step, data in enumerate(train_loader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = model.HC_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if show_iters > 0:
                if step % show_iters == (show_iters - 1):
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, step + 1,  running_loss / show_iters))
                    running_loss = 0.0

        if model_save_epoch > 0:
            if (epoch + 1) % model_save_epoch == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(model_save_dir, 'checkpoint_' + str(epoch) + '.tar'))

        with torch.no_grad():
            print('*' * 10, 'Begin to validation', '*' * 10)
            if 'val' in loaders:
                val_loader = loaders['val']
                top_k_right = 0.0
                count = 0.0
                for _, val_data in enumerate(val_loader, 0):
                    images, labels = val_data
                    count += len(labels)
                    images = images.to(device)
                    outputs = model.HC_prediction(images)
                    outputs = outputs.to('cpu').numpy()
                    predictions = outputs.argsort(axis=1)
                    predictions = predictions[:, -3:]
                    top_k_right += Top_K_Right(labels.numpy(), predictions)
                top_k_acc = top_k_right * 1.0 / count
                epoch_result['val_result'] = top_k_acc
                print('top 3 acc:\t', top_k_acc)

            print('*' * 10, 'Begin to predict test result', '*' * 10)
            if 'test' in loaders:
                prediction_epoch = []
                test_loader = loaders['test']
                for _, test_data in enumerate(test_loader, 0):
                    images = test_data.to(device)
                    outputs = model.HC_prediction(images)
                    outputs = outputs.to('cpu').numpy()
                    predictions = outputs.argsort(axis=1)
                    predictions = predictions[:, -3:]
                    prediction_epoch.extend(predictions.tolist())
                epoch_result['test_predictions'] = prediction_epoch

        pickle.dump(epoch_result, open(log_file_name_prefix + '_' + str(epoch) + '.pkl', 'wb'))
        results.append(epoch_result)

    pickle.dump(results, open(log_file_name_prefix + '_all.pkl', 'wb'))