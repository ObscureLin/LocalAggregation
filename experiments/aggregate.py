import sys

sys.path.append("../")

from torch.utils.data import DataLoader
from utils.data_loader import DDRDataset, transform_valid
from models.vgg import *

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clients_num = 3
    clients_weights = []
    agg_model = model = VGGnet(in_channels=3, num_classes=6).to(device)
    for i in range(clients_num):
        client_model = VGGnet(in_channels=3, num_classes=6).to(device)
        client_model.load_state_dict(torch.load("../outputs/client" + str(i) + "_best_model.pt"))
        # client_model.load_state_dict(copy.deepcopy(torch.load("../outputs/client" + str(i) + "_best_model.pt", device)()))
        clients_weights.append(client_model.state_dict())

    for key, var in agg_model.state_dict().items():
        sum_weight = torch.zeros_like(clients_weights[0][key].size())
        for weights in clients_weights:
            temp_weight = weights[key]
            sum_weight += (temp_weight / clients_num)
        agg_model.state_dict()[key] = sum_weight

    # valid model
    valid_dataset = DDRDataset("../dataset/DR_grading", dataset_type="valid", transforms=transform_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=12)

    # validation
    model.eval()
    correct = 0
    total = 0
    val_acc_list = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    acc_val = correct / total
    val_acc_list.append(acc_val)

    print("acc_val = {}".format(acc_val))
