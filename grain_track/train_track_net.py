import os
import time
import PIL.Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython.display import clear_output
from grain_track.utility import *
from grain_track.modified_models.vgg import vgg13_bn
from grain_track.modified_models.densenet import densenet161
from grain_track.modified_models.grain_track_datasets import GrainTrackDatasets, RandomChannelFlip


data = "simulated"  # or "real"
cnn_model_index = 0
if cnn_model_index == 0:
    experiment_name = data + '_vgg13_bn'
elif cnn_model_index == 1:
    experiment_name = data + '_densenet161'
device = 'cuda:1'
if data == "real":
    loss_weight = torch.Tensor([3.41, 1.0]).to(device)
elif data == "simulated":
    loss_weight = torch.Tensor([2.43, 1.0]).to(device)


def plot(epoch, train_loss_list, val_loss_list, train_acc_list, val_acc_list):
    clear_output(True)
    print('Epoch %s. train loss: %s. val loss: %s.' % (epoch, train_loss_list[-1], val_loss_list[-1]))
    print('Epoch %s. train acc: %s. val acc: %s.' % (epoch, train_acc_list[-1], val_acc_list[-1]))
    print('Best val acc: %s.' % (max(val_acc_list)))
    plt.figure()
    plt.plot(train_loss_list, color="r", label="train loss")
    plt.plot(val_loss_list, color="b", label="val loss")
    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss " + experiment_name,fontsize=16)
    plt.savefig(os.path.join(figure_dir, experiment_name + '_loss.png'))
    plt.figure()
    plt.plot(train_acc_list, color="r", label="train acc")
    plt.plot(val_acc_list, color="b", label="val acc")
    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.ylabel("Acc")
    plt.title("Predicted accuracy "+ experiment_name,fontsize=16)
    plt.savefig(os.path.join(figure_dir, experiment_name + '_acc.png'))
    plt.show()


def write_log(content):
    print(content)
    with open(experiment_name + ".txt",'a') as fp:
        fp.write(content+"\r\n")

# 参数
learning_rate = 1e-4
momentum = 0.9
epochs = 20
batch_size = 40
display_step = 500
shuffle = True
num_classes = 2

best_acc = 0.0
loss_train = []  # 训练集loss
acc_train = []   # 训练集正确率
loss_val = []    # 验证集loss
acc_val = []     # 验证集正确率



# 数据集
project_address = os.getcwd()
figure_dir = os.path.join(project_address, "figure")
parameter_dir = os.path.join(project_address, "parameter")
make_out_dir(figure_dir)
make_out_dir(parameter_dir)
data_dir = os.path.join(project_address, "datasets", "net_train", data)
train_data_transforms = transforms.Compose([
    transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
    RandomChannelFlip(),
    transforms.RandomChoice([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]),
    transforms.RandomChoice([
        transforms.RandomRotation((90, 90), expand=True),
        transforms.RandomRotation((180, 180), expand=True),
        transforms.RandomRotation((270, 270), expand=True),
    ]),
    transforms.ToTensor(),
])
val_data_transforms = transforms.Compose([
    transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor(),
])

image_datasets = {}
image_datasets['train'] = GrainTrackDatasets(os.path.join(data_dir, 'train'), train_data_transforms)
image_datasets['val'] = GrainTrackDatasets(os.path.join(data_dir, 'val'), val_data_transforms)
# image_datasets = {x: GrainTrackDatasets(os.path.join(data_dir, x), data_transforms) for x in ['train', 'val']}
dataloders = {}
dataloders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,  shuffle=shuffle, num_workers=1)
dataloders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size-20,  shuffle=shuffle, num_workers=1)

# dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,  shuffle=shuffle, num_workers=1) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

write_log('The number of images in train: {}'.format(dataset_sizes['train']))
write_log('The number of images in val: {}'.format(dataset_sizes['val']))
write_log('loss_weight:{}'.format(loss_weight))

# 网络模型
if cnn_model_index == 0:
    model = vgg13_bn()
    model.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                     nn.ReLU(True),
                                     nn.Dropout(),
                                     nn.Linear(4096, 4096),
                                     nn.ReLU(True),
                                     nn.Dropout(),
                                     nn.Linear(4096, num_classes))
elif cnn_model_index == 1:
    model = densenet161()
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# 定义损失函数，这里采用交叉熵函数
loss_fn = nn.CrossEntropyLoss(weight=loss_weight)

optimizer = optim.RMSprop(model.parameters(), learning_rate, momentum)

since = time.time()
best_model_wts = model.state_dict()


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by half every 10 epochs until 1e-5"""
    lr = learning_rate * (0.8 ** (epoch // 1))
    if not lr < 1e-6:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


for epoch in range(epochs):
    write_log('Epoch [{}/{}]:'.format(epoch + 1, epochs))

    # 每一轮都跑一遍训练集和验证集
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # 把module设成training模式，对Dropout和BatchNorm有影响
        else:
            model.eval()  # 把module设置为评估模式

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for i, data in enumerate(dataloders[phase]):
            # get the inputs
            inputs = data['image'].to(device)
            labels = data['label'].to(device)
            # 先将网络中的所有梯度置0
            optimizer.zero_grad()

            # 网络的前向传播
            outputs = model(inputs)
            # 计算损失
            loss = loss_fn(outputs, labels)
            _, preds = torch.max(outputs.data, 1)

            # 记录当前batch_size的loss以及数据对应的分类准确数量
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

            if i % display_step == 0:
                write_log('\t{} {}-{}: Loss: {:.4f} Acc: {:.4f}%'.format(phase, epoch + 1, i, loss.item() / batch_size,
                                                                         torch.sum(
                                                                             preds == labels.data).item() / batch_size * 100))

            # 训练时，应用回传和优化
            if phase == 'train':
                loss.backward()
                optimizer.step()

        # 计算并打印这一轮训练的loss和分类准确率
        if phase == 'train':
            epoch_loss_train = running_loss / dataset_sizes['train']
            epoch_acc_train = running_corrects.item() / dataset_sizes['train']
            loss_train.append(epoch_loss_train)
            acc_train.append(epoch_acc_train)
        else:
            epoch_loss_val = running_loss / dataset_sizes['val']
            epoch_acc_val = running_corrects.item() / dataset_sizes['val']
            loss_val.append(epoch_loss_val)
            acc_val.append(epoch_acc_val)

        if phase == 'val':
            write_log('\ttrain Loss: {:.6f} Acc: {:.8f}%'.format(epoch_loss_train, epoch_acc_train * 100))
            write_log('\tvalidation Loss: {:.6f} Acc: {:.8f}%'.format(epoch_loss_val, epoch_acc_val * 100))

        # deep copy the model
        if phase == 'val' and epoch_acc_val > best_acc:
            best_acc = epoch_acc_val
            best_model_wts = model.state_dict()
            write_log("Updating")
            torch.save(best_model_wts, os.path.join(parameter_dir, experiment_name + '.pkl'))
    plot(epoch, loss_train, loss_val, acc_train, acc_val)

    torch.save(model.state_dict(), os.path.join(parameter_dir, experiment_name + "_" + str(epoch) + '.pkl'))
    time_elapsed = time.time() - since
    write_log('Time passed {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60,
                                                           time_elapsed % 60))
    write_log('-' * 20)

# 计算训练所耗时间
time_elapsed = time.time() - since
write_log('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60,
                                                                time_elapsed % 60))
write_log('Best validation Acc: {:4f}'.format(best_acc))
print('loss_train: ' + str(loss_train))
print('loss_val: ' + str(loss_val))
print('acc_train: ' + str(acc_train))
print('acc_val: ' + str(acc_val))
