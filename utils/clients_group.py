import os.path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.data_loader import DDRDataset, transform_valid, read_txt, SubDataset


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    参数为alpha的Dirichlet分布将数据索引划分为n_clients个子集
    '''
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]
    # 记录每个K个类别对应的样本下标

    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


class Client(object):
    def __init__(self, *, client_id, model, train_dataloader, valid_dataloader, device, outputs_dir):
        self.client_id = client_id
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = device
        self.local_weights = None  # model.state_dict()
        self.outputs_dir = outputs_dir
        self.model = self.model.to(self.device)

    def train_client(self):
        # prepare hyper parameters
        criterion = nn.CrossEntropyLoss().to(self.device)
        learning_rate = 1e-4
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

        # train
        val_acc_list = []
        for epoch in range(20):
            self.model.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                print("[INFO] Client:{} Train_Batch {} :The train_loss is : {}.".format(self.client_id, batch_idx,
                                                                                        loss.item()))
            # validation
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.valid_dataloader):
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    _, predicted = torch.max(output.data, dim=1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            acc_val = correct / total
            val_acc_list.append(acc_val)

            # save model
            torch.save(self.model.state_dict(),
                       os.path.join(self.outputs_dir, "client" + str(self.client_id), "_last_model.pt"))
            if acc_val == max(val_acc_list):
                torch.save(self.model.state_dict(),
                           os.path.join(self.outputs_dir, "client" + str(self.client_id), "_best_model.pt"))
                self.local_weights = self.model.state_dict()
                print("[INFO] Client:{} Save epoch {} model".format(self.client_id, epoch))

            print(
                "[INFO] Client:{} Epoch = {},  total_loss = {},  acc_val = {}".format(self.client_id, epoch, train_loss,
                                                                                      acc_val))

    def get_local_weights(self):
        return self.local_weights

    def update_local_weights(self, new_weights):
        self.model = new_weights
        self.local_weights = new_weights


class ClientsGroup(object):
    def __init__(self, *, model, clients_num, dataset_path, device, outputs_dir, isIID="NO"):
        self.model = model
        self.clients_num = clients_num
        self.dataset_path = dataset_path  # ./dataset/DR_grading
        self.device = device
        self.outputs_dir = outputs_dir
        self.isIID = isIID
        self.clients = []  # Client[]
        self.aggregated_weights = None
        valid_dataset = DDRDataset(self.dataset_path, dataset_type="valid", transforms=transform_valid)
        valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=12)  #
        self.train_images, self.train_labels = read_txt(os.path.join(self.dataset_path, "train.txt"))

        for index in range(len(self.train_images)):
            self.train_images[index] = os.path.join(dataset_path, "train", self.train_images[index])

        rand_num = random.randint(0, 100)
        random.seed(rand_num)
        random.shuffle(self.train_images)
        random.seed(rand_num)
        random.shuffle(self.train_labels)

        if self.isIID == "NO":
            shard_size = len(self.train_labels) // self.clients_num // 2
            shards_id = np.random.permutation(len(self.train_labels) // shard_size)
            print("*" * 20)
            print(len(self.train_labels))
            print(shard_size)
            print(shards_id)
            print(shards_id.shape)
            print(self.train_labels)
            print("*" * 20)
            for i in range(self.clients_num):
                # 0 2 4 6...... 偶数
                shards_id1 = shards_id[i * 2]
                # 0+1 = 1 2+1 = 3 .... 奇数
                shards_id2 = shards_id[i * 2 + 1]
                # 将数据以及的标签分配给该客户端
                data_shards1 = self.train_images[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
                data_shards2 = self.train_images[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
                label_shards1 = self.train_labels[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
                label_shards2 = self.train_labels[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
                # local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack(
                #    (label_shards1, label_shards2))
                # local_label = np.argmax(local_label, axis=1)
                sub_data = data_shards1 + data_shards2
                sub_label = label_shards1 + label_shards2
                print("===============================")
                print(sub_label)
                print(len(sub_label))
                print("===============================")
                sub_dataset = SubDataset(image_list=sub_data,
                                         label_list=sub_label)
                sub_dataloader = DataLoader(sub_dataset, batch_size=32, shuffle=True, num_workers=12)
                self.clients.append(
                    Client(client_id=i, model=self.model,
                           train_dataloader=sub_dataloader,
                           valid_dataloader=valid_dataloader, device=self.device, outputs_dir=self.outputs_dir))

    def train_all_clients(self):
        for client in self.clients:
            client.train_client()

    def aggregate_weights(self):
        pass

    def valid_aggregated_weights(self):
        pass
