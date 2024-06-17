import torch
import argparse
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryConfusionMatrix
from effdet import  DetBenchPredict

from src.data.dataset import XrayDataset
from src.data.transforms import get_val_transforms
from src.models.networks import create_model


def load_from_checkpoint(
    model,
    checkpoint
):
    sd = torch.load(checkpoint)['state_dict']
    new_sd = OrderedDict()
    for k, v in sd.items():
        new_sd[k[8:]] = v
    model.load_state_dict(new_sd, strict=True)
    model.eval()
    return model


def evaluate(
    model,
    dataloader,
    device,
    confusion_matrix_th=0.4,
):
    """Evaluate test dataset function.

    Args:
        model: Your pretrained model
        dataloader: pytorch dataloader with images and gt targets
        device: where do you want to infer model, on gpu or cpu
        confusion_matrix_th: Threshhold for metrics calculation. Defaults to 0.3.

        Prints Sensitivity and Specificity metrics to the terminal

    """
    confusion_matrix = BinaryConfusionMatrix()
    predictions = []
    with torch.no_grad():
        for image, targets in dataloader:
            model = model.to(device)
            outputs = model(image.float().to(device))
            for i in range(image.shape[0]):
                boxes = outputs[i].detach().cpu().numpy()[:, :4]
                scores = outputs[i].detach().cpu().numpy()[:, 4]
                labels = outputs[i].detach().cpu().numpy()[:, 5]
                max_index = np.argmax(scores)
                boxes = boxes[max_index].reshape(1, -1)
                scores = scores[max_index].reshape(1, -1)
                labels = labels[max_index].reshape(1, -1)

                boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

                predictions.append({
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels,
                })
                preds_bin = (scores > confusion_matrix_th).astype(int)
                targets_bin = (targets['cls'].numpy() > 0).astype(int)
                confusion_matrix.update(torch.tensor(preds_bin), torch.tensor(targets_bin))
            # Compute metrics
        conf_matrix = confusion_matrix.compute()
        tn, fp, fn, tp = conf_matrix.flatten()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        print(f'Sensitivity: {sensitivity}')
        print(f'Specificity: {specificity}')

def main(args):
    image_folder_path = args.image_folder_path
    annotations_file = args.annotations_file
    checkpoint = args.checkpoint
    th = args.cm_threshold

    eval_dataset = XrayDataset(image_folder_path, annotations_file, transforms=get_val_transforms())
    evaloader = DataLoader(eval_dataset, batch_size=1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net, _ = create_model(
        num_classes=args.num_classes,
        image_size=args.image_size,
        architecture=args.architecture,
        backbone=args.backbone,
        train=False,
    )
    model = DetBenchPredict(net)
    model = load_from_checkpoint(model, checkpoint)

    evaluate(model, evaloader, device, th)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate X-ray model')
    parser.add_argument('--image_folder_path', type=str, default='/data1/xray_data/images', help='Path to the folder containing images')
    parser.add_argument('--annotations_file', type=str, default='dataset/val_df.csv', help='Path to the annotations CSV file')
    parser.add_argument('--checkpoint', type=str, default='/srv/checkpoints/xray/2024-06-16 21:43:41/best.ckpt', help='Path to the model checkpoint')
    parser.add_argument('--cm_threshold', type=float, default=0.3, help='Threshold for confusion matrix predictions')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes')
    parser.add_argument('--image_size', type=int, default=512, help='Image size')
    parser.add_argument('--architecture', type=str, default='tf_efficientdet_lite3', help='Model architecture')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0', help='Backbone model')

    args = parser.parse_args()
    main(args)
