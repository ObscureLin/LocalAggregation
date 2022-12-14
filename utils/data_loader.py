from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision import transforms as T

MEAN = (0, 0, 0)
STD = (1, 1, 1)

transform_train = T.Compose([
    T.Resize([512, 512]),
    T.RandomCrop(224, padding=4),  # 学姐使用的 448 vgg16 默认224
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize(MEAN, STD)])

transform_valid = T.Compose([
    T.Resize([224, 224]),
    T.ToTensor(),
    T.Normalize(MEAN, STD)
])

transform_test = T.Compose([
    T.Resize([224, 224]),
    T.ToTensor(),
    T.Normalize(MEAN, STD)
])


# we randomly selected half of the original images as a training set, 2/10 as a validation set, and 3/10 as a testing set.
def read_txt(path):
    # 读取txt文件，将图像路径和标签写入到列表中并返回
    ims, labels = [], []
    with open(path, 'r') as f:
        for sample in f.readlines():
            im, label = sample.strip().split(" ")
            ims.append(im)
            labels.append(int(label))
    return ims, labels


class SubDataset(Dataset):
    def __init__(self, *, image_list, label_list):
        super().__init__()
        self.image_list = image_list
        self.label_list = label_list

    def __getitem__(self, index):
        image, label = self.image_list[index], self.label_list[index]
        image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = transform_train(image)
        return image, label

    def __len__(self):
        return len(self.label_list)


class DDRDataset(Dataset):
    def __init__(self, dataset_path="../dataset/DR_grading", *, dataset_type="train", transforms=None):
        super().__init__()
        self.dataset_type = dataset_type
        self.dataset_path = dataset_path
        # store the path of the images and the text of the labels
        self.images, self.labels = read_txt(os.path.join(self.dataset_path, self.dataset_type + ".txt"))
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            self.transforms = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = os.path.join(self.dataset_path, self.dataset_type, image)
        image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    train_dataset = DDRDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for batch_index, (data, label) in enumerate(train_dataloader):
        print(batch_index, type(data), data.shape, type(label))
