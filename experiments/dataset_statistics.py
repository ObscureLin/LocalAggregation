import sys

sys.path.append("../")
from utils.data_loader import *

if __name__ == "__main__":
    # initialize the statistics list
    train_set_statistics = [0, 0, 0, 0, 0, 0]
    valid_set_statistics = [0, 0, 0, 0, 0, 0]
    test_set_statistics = [0, 0, 0, 0, 0, 0]
    total_statistics = [0, 0, 0, 0, 0, 0]

    # initialize train valid test dataset
    batch_size = 1  # we recommend use 32
    train_dataset = DDRDataset("../dataset/DR_grading", dataset_type="train")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("[INFO] The length of the train data is {}.".format(train_dataset.__len__()))

    valid_dataset = DDRDataset("../dataset/DR_grading", dataset_type="valid")
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    print("[INFO] The length of the valid data is {}.".format(valid_dataset.__len__()))

    test_dataset = DDRDataset("../dataset/DR_grading", dataset_type="test")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print("[INFO] The length of the test  data is {}.".format(test_dataset.__len__()))

    for batch_index, (data, label) in enumerate(train_dataloader):
        # print("[VERBOSE] {}".format(batch_index))
        _class = label.numpy().tolist()[0]
        train_set_statistics[_class] = train_set_statistics[_class] + 1
    print(
        "[INFO] The train data set has class 0 : {} class 1 : {} class 2 : {} class 3 : {} class 4 : {} class 5 : {} total : {} ".format(
            train_set_statistics[0], train_set_statistics[1], train_set_statistics[2], train_set_statistics[3],
            train_set_statistics[4], train_set_statistics[5], train_dataset.__len__()))

    for batch_index, (data, label) in enumerate(valid_dataloader):
        # print("[VERBOSE] {}".format(batch_index))
        _class = label.numpy().tolist()[0]
        valid_set_statistics[_class] = valid_set_statistics[_class] + 1
    print(
        "[INFO] The valid data set has class 0 : {} class 1 : {} class 2 : {} class 3 : {} class 4 : {} class 5 : {} total : {} ".format(
            valid_set_statistics[0], valid_set_statistics[1], valid_set_statistics[2], valid_set_statistics[3],
            valid_set_statistics[4], valid_set_statistics[5], valid_dataset.__len__()))

    for batch_index, (data, label) in enumerate(test_dataloader):
        # print("[VERBOSE] {}".format(batch_index))
        _class = label.numpy().tolist()[0]
        test_set_statistics[_class] = test_set_statistics[_class] + 1
    print(
        "[INFO] The test data set has class 0 : {} class 1 : {} class 2 : {} class 3 : {} class 4 : {} class 5 : {} total : {} ".format(
            test_set_statistics[0], test_set_statistics[1], test_set_statistics[2], test_set_statistics[3],
            test_set_statistics[4], test_set_statistics[5], test_dataset.__len__()))

    print("--" * 15)
