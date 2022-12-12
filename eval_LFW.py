import torch
import torch.backends.cudnn as cudnn

from nets.facenet import Facenet
from utils.dataloader import LFWDataset
from utils.utils_metrics import test

if __name__ == "__main__":

    cuda            = True
    backbone        = "inception_resnetv1" # inception_resnetv1
    input_shape     = [160, 160, 3]

    model_path      = "model_data/facenet_mobilenet.pth"
    model_path      = "/home/wangcong/Course/FaceRecognition/logs/ep100-loss0.141-val_loss1.208.pth"
    model_path      = "/home/wangcong/Course/FaceRecognition/logs/ep062-loss0.086-val_loss0.738.pth"

    


    lfw_dir_path    = "lfw"
    lfw_pairs_path  = "model_data/lfw_pair.txt"

    batch_size      = 256*10
    log_interval    = 1

    png_save_path   = "model_data/roc_test.png"

    test_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=batch_size, shuffle=False)

    model = Facenet(backbone=backbone, mode="predict")

    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model  = model.eval()

    if cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()

    test(test_loader, model, png_save_path, log_interval, batch_size, cuda)
