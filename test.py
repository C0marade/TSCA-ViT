import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_osteosarcoma import Synapse_dataset
from networks.TSCAFormer import DAEFormer
from utils import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument(
    "--volume_path",
    type=str,
    default="/images/PublicDataset/Transunet_synaps/project_TransUNet/data/Synapse/",
    help="root dir for validation volume data",
)
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument("--num_classes", type=int, default=9, help="output channel of network")
parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse", help="list dir")
parser.add_argument("--output_dir", type=str, default="./model_out", help="output dir")
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
parser.add_argument("--is_savenii", action="store_true", help="whether to save results during inference")
parser.add_argument("--test_save_dir", type=str, default="../predictions", help="saving prediction as nii!")
parser.add_argument("--deterministic", type=int, default=1, help="whether to use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.05, help="segmentation network learning rate")
parser.add_argument("--seed", type=int, default=1234, help="random seed")

args = parser.parse_args()

if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")


def dice_coefficient(true, pred, classes):
    dice = []
    for i in range(classes):
        true_class = (true == i).astype(np.float32)
        pred_class = (pred == i).astype(np.float32)
        intersection = np.sum(true_class * pred_class)
        if np.sum(true_class) + np.sum(pred_class) == 0:
            dice.append(1.0)  # Perfect match if no true/pred pixels
        else:
            dice.append((2.0 * intersection) / (np.sum(true_class) + np.sum(pred_class)))
    return np.mean(dice)


def iou_score(true, pred, classes):
    iou = []
    for i in range(classes):
        true_class = (true == i).astype(np.float32)
        pred_class = (pred == i).astype(np.float32)
        intersection = np.sum(true_class * pred_class)
        union = np.sum(true_class) + np.sum(pred_class) - intersection
        if union == 0:
            iou.append(1.0)  # Perfect match if no union
        else:
            iou.append(intersection / union)
    return np.mean(iou)


def test_single_volume(image, label, model, classes, patch_size, test_save_path=None, case=None, z_spacing=1):
    model.eval()
    with torch.no_grad():
        # Forward pass
        image = image.cuda()
        outputs = model(image)
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

        # Flatten tensors for metric computation
        pred_flat = outputs.cpu().numpy().flatten()
        label_flat = label.numpy().flatten()

        # Compute metrics
        accuracy = np.mean(pred_flat == label_flat)
        precision = precision_score(label_flat, pred_flat, average="macro", zero_division=0)
        recall = recall_score(label_flat, pred_flat, average="macro", zero_division=0)
        dice = dice_coefficient(label_flat, pred_flat, classes)
        iou = iou_score(label_flat, pred_flat, classes)

        logging.info(f"Case {case}: Accuracy={accuracy}, Precision={precision}, Recall={recall}, DSC={dice}, IoU={iou}")

        return accuracy, precision, recall, dice, iou


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", img_size=args.img_size, list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()

    metric_totals = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "dice": 0.0, "iou": 0.0}
    sample_count = len(testloader)

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch["case_name"][0]

        # Compute metrics for each sample
        accuracy, precision, recall, dice, iou = test_single_volume(
            image,
            label,
            model,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
            z_spacing=args.z_spacing,
        )

        # Accumulate metrics
        metric_totals["accuracy"] += accuracy
        metric_totals["precision"] += precision
        metric_totals["recall"] += recall
        metric_totals["dice"] += dice
        metric_totals["iou"] += iou

    # Compute averages
    for key in metric_totals.keys():
        metric_totals[key] /= sample_count

    # Log final performance
    logging.info("Final Test Metrics:")
    for key, value in metric_totals.items():
        logging.info(f"{key.capitalize()}: {value:.4f}")

    return metric_totals


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        "Synapse": {
            "Dataset": Synapse_dataset,
            "z_spacing": 1,
        },
    }
    dataset_name = args.dataset
    args.Dataset = dataset_config[dataset_name]["Dataset"]
    args.z_spacing = dataset_config[dataset_name]["z_spacing"]

    net = DAEFormer(num_classes=args.num_classes).cuda(0)

    snapshot = os.path.join(args.output_dir, "best_model.pth")
    if not os.path.exists(snapshot):
        snapshot = snapshot.replace("best_model", "transfilm_epoch_" + str(args.max_epochs - 1))
    msg = net.load_state_dict(torch.load(snapshot))
    print("self trained swin unet", msg)
    snapshot_name = snapshot.split("/")[-1]

    log_folder = "./test_log/test_log_"
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=log_folder + "/" + snapshot_name + ".txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)
