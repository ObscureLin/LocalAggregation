import sys

sys.path.append("../")
from utils.clients_group import *
from models.vgg import *

if __name__ == "__main__":
    # initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] The device used by torch is {}.".format(device))
    model = VGGnet(in_channels=3, num_classes=6)
    clients_group = ClientsGroup(model=model, clients_num=3, dataset_path="../dataset/DR_grading", device=device,
                                 outputs_dir="../outputs", isIID="NO")
    clients_group.train_all_clients()
