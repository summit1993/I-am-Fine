import os
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
import pickle
from torch.optim.lr_scheduler import MultiStepLR
from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, save_dir
from core import model, my_dataset
from core.utils import init_log, progress_bar

data_dir = "/home1/CVPR"

os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,11,1'
start_epoch = 1
save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)
logging = init_log(save_dir)
_print = logging.info

# read dataset
train_data = pickle.load(open(os.path.join(data_dir, 'data/train.pkl'), 'rb'))
trainset = my_dataset.MyDataset(train_data['images'],  train_data['labels'],
                                os.path.join(data_dir, 'images/train'), 'train')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=8, drop_last=False)
val_data = pickle.load(open(os.path.join(data_dir, 'data/val.pkl'), 'rb'))
valset = my_dataset.MyDataset(val_data['images'],  val_data['labels'],
                               os.path.join(data_dir, 'images/val'), 'inference')
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=8, drop_last=False)
test_data = pickle.load(open(os.path.join(data_dir, 'data/test.pkl'), 'rb'))
testset = my_dataset.MyDataset(test_data['images'],  None,
                               os.path.join(data_dir, 'images/test'), 'inference')
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=8, drop_last=False)
# define model
net = model.attention_net(topN=PROPOSAL_NUM)
if resume:
    ckpt = torch.load(resume)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1
creterion = torch.nn.CrossEntropyLoss()

# define optimizers
raw_parameters = list(net.pretrained_model.parameters())
part_parameters = list(net.proposal_net.parameters())
concat_parameters = list(net.concat_net.parameters())
partcls_parameters = list(net.partcls_net.parameters())

raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)
concat_optimizer = torch.optim.SGD(concat_parameters, lr=LR, momentum=0.9, weight_decay=WD)
part_optimizer = torch.optim.SGD(part_parameters, lr=LR, momentum=0.9, weight_decay=WD)
partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=LR, momentum=0.9, weight_decay=WD)
schedulers = [MultiStepLR(raw_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(concat_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(part_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(partcls_optimizer, milestones=[60, 100], gamma=0.1)]
net = net.cuda()
net = DataParallel(net)

for epoch in range(start_epoch, 500):
    for scheduler in schedulers:
        scheduler.step()

    # begin training
    _print('--' * 50)
    net.train()
    for i, data in enumerate(trainloader):
        img, label = data[0].cuda(), data[1].cuda()
        batch_size = img.size(0)
        raw_optimizer.zero_grad()
        part_optimizer.zero_grad()
        concat_optimizer.zero_grad()
        partcls_optimizer.zero_grad()

        raw_logits, concat_logits, part_logits, _, top_n_prob = net(img)
        part_loss = model.list_loss(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                    label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(batch_size, PROPOSAL_NUM)
        raw_loss = creterion(raw_logits, label)
        concat_loss = creterion(concat_logits, label)
        rank_loss = model.ranking_loss(top_n_prob, part_loss)
        partcls_loss = creterion(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                 label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))

        total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
        total_loss.backward()
        raw_optimizer.step()
        part_optimizer.step()
        concat_optimizer.step()
        partcls_optimizer.step()
        progress_bar(i, len(trainloader), 'train')

    if epoch % SAVE_FREQ == 0:
        # evaluate on val set
        val_loss = 0
        val_correct = 0
        total = 0
        for i, data in enumerate(valloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                _, concat_logits, _, _, _ = net(img)
                # calculate loss
                concat_loss = creterion(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                val_correct += torch.sum(concat_predict.data == label.data)
                val_loss += concat_loss.item() * batch_size
                progress_bar(i, len(valloader), 'eval val set')

        val_acc = float(val_correct) / total
        val_loss = val_loss / total
        _print(
            'epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample: {}'.format(
                epoch,
                val_loss,
                val_acc,
                total))

        # prediction on test set
        print('begin to predict')
        epoch_result = {}
        prediction_epoch = []
        for i, data in enumerate(testloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                _, concat_logits, _, _, _ = net(img)
                outputs = concat_logits.to('cpu').numpy()
                predictions = outputs.argsort(axis=1)
                predictions = predictions[:, -3:]
                prediction_epoch.extend(predictions.tolist())
        epoch_result['test_predictions'] = prediction_epoch
        pickle.dump(epoch_result, open(save_dir + '_test_predict_' + str(epoch) + '.pkl', 'wb'))

        # save model
        net_state_dict = net.module.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save({
            'epoch': epoch,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'net_state_dict': net_state_dict},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

print('finishing training')
