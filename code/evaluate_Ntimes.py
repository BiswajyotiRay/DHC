import os
import numpy as np
import argparse
from medpy import metric
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp', type=str, default="fully")
parser.add_argument('--folds', type=int, default=3)
parser.add_argument('--cps', type=str, default=None)
args = parser.parse_args()

from utils import read_list, read_nifti
import torch
import torch.nn.functional as F
from utils.config import Config
from utils import read_nifti
import numpy as np # Ensure this import is present if pred is returned by read_nifti

config = Config(args.task)

def resize_array(arr, target_shape):
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions
    resized_tensor = F.interpolate(tensor, size=target_shape, mode='nearest').squeeze()
    return resized_tensor.numpy()

if __name__ == '__main__':

    ids_list = read_list('test', task=args.task)
    results_all_folds = []

    txt_path = "./logs/" + args.exp + "/evaluation_res.txt"
    print("\n Evaluating...")
    fw = open(txt_path, 'w')
    for fold in range(1, args.folds + 1):

        test_cls = [i for i in range(1, config.num_cls)]
        values = np.zeros((len(ids_list), len(test_cls), 2))  # dice and asd

        for idx, data_id in enumerate(tqdm(ids_list)):
            file_path = os.path.join("./logs", args.exp, "fold" + str(fold), "predictions_" + str(args.cps), f'{data_id}.nii.gz')
            print(f"Checking file path: {file_path}")  # Debugging print statement

            if not os.path.exists(file_path):
                print(f"File does not exist: {file_path}")
                continue  # Skip this iteration if the file does not exist

            pred = read_nifti(file_path)
            if args.task == "amos":
                label_path = os.path.join(config.base_dir, 'labelsVa', f'{data_id}.nii.gz')
            else:
                label_path = os.path.join(config.base_dir, 'labelsTr', f'label{data_id}.nii.gz')

            print(f"Checking label path: {label_path}")  # Debugging print statement

            if not os.path.exists(label_path):
                print(f"Label file does not exist: {label_path}")
                continue  # Skip this iteration if the label file does not exist

            label = read_nifti(label_path)
            label = label.astype(np.int8)

            # Resize label to match pred shape
            pred_shape = pred.shape
            label = resize_array(label, pred_shape)
            label = label.astype(np.int8)  # Ensure label is integer type

            for i in test_cls:
                pred_i = (pred == i)
                label_i = (label == i)
                if pred_i.sum() > 0 and label_i.sum() > 0:
                    dice = metric.binary.dc(pred_i, label_i) * 100
                    hd95 = metric.binary.asd(pred_i, label_i)
                    values[idx][i - 1] = np.array([dice, hd95])
                elif pred_i.sum() > 0 and label_i.sum() == 0:
                    dice, hd95 = 0, 128
                elif pred_i.sum() == 0 and label_i.sum() > 0:
                    dice, hd95 = 0, 128
                elif pred_i.sum() == 0 and label_i.sum() == 0:
                    dice, hd95 = 1, 0

                values[idx][i - 1] = np.array([dice, hd95])

            # Print intermediate Dice scores for debugging
            print(f"Intermediate Dice scores for data_id {data_id}: {values[idx]}")

        values_mean_cases = np.mean(values, axis=0)
        results_all_folds.append(values)
        fw.write("Fold" + str(fold) + '\n')
        fw.write("------ Dice ------" + '\n')
        fw.write(str(np.round(values_mean_cases[:, 0], 1)) + '\n')
        fw.write("------ ASD ------" + '\n')
        fw.write(str(np.round(values_mean_cases[:, 1], 1)) + '\n')
        fw.write('Average Dice:' + str(np.mean(values_mean_cases, axis=0)[0]) + '\n')
        fw.write('Average  ASD:' + str(np.mean(values_mean_cases, axis=0)[1]) + '\n')
        fw.write("=================================")
        print("Fold", fold)
        print("------ Dice ------")
        print(np.round(values_mean_cases[:, 0], 1))
        print("------ ASD ------")
        print(np.round(values_mean_cases[:, 1], 1))
        print(np.mean(values_mean_cases, axis=0)[0], np.mean(values_mean_cases, axis=0)[1])

    results_all_folds = np.array(results_all_folds)

    fw.write('\n\n\n')
    fw.write('All folds' + '\n')

    results_folds_mean = results_all_folds.mean(0)

    for i in range(results_folds_mean.shape[0]):
        fw.write("=" * 5 + " Case-" + str(ids_list[i]) + '\n')
        fw.write('\tDice:' + str(np.round(results_folds_mean[i][:, 0], 2).tolist()) + '\n')
        fw.write('\t ASD:' + str(np.round(results_folds_mean[i][:, 1], 2).tolist()) + '\n')
        fw.write('\t' + 'Average Dice:' + str(np.mean(results_folds_mean[i], axis=0)[0]) + '\n')
        fw.write('\t' + 'Average  ASD:' + str(np.mean(results_folds_mean[i], axis=0)[1]) + '\n')

    fw.write("=================================\n")
    fw.write('Final Dice of each class\n')
    fw.write(str([round(x, 1) for x in results_folds_mean.mean(0)[:, 0].tolist()]) + '\n')
    fw.write('Final ASD of each class\n')
    fw.write(str([round(x, 1) for x in results_folds_mean.mean(0)[:, 1].tolist()]) + '\n')
    print("=================================")
    print('Final Dice of each class')
    print(str([round(x, 1) for x in results_folds_mean.mean(0)[:, 0].tolist()]))
    print('Final ASD of each class')
    print(str([round(x, 1) for x in results_folds_mean.mean(0)[:, 1].tolist()]))
    std_dice = np.std(results_all_folds.mean(1).mean(1)[:, 0])
    std_hd = np.std(results_all_folds.mean(1).mean(1)[:, 1])

    fw.write('Final Avg Dice: ' + str(round(results_folds_mean.mean(0).mean(0)[0], 2)) + '±' + str(round(std_dice, 2)) + '\n')
    fw.write('Final Avg  ASD: ' + str(round(results_folds_mean.mean(0).mean(0)[1], 2)) + '±' + str(round(std_hd, 2)) + '\n')

    print('Final Avg Dice: ' + str(round(results_folds_mean.mean(0).mean(0)[0], 2)) + '±' + str(round(std_dice, 2)))
    print('Final Avg  ASD: ' + str(round(results_folds_mean.mean(0).mean(0)[1], 2)) + '±' + str(round(std_hd, 2)))
