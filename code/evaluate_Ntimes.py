import os
import numpy as np
import argparse
from medpy import metric
from tqdm import tqdm
import torch
import torch.nn.functional as F

from utils import read_list, read_nifti
from utils.config import Config

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp', type=str, default="unetrpp")
parser.add_argument('--folds', type=int, default=3)
parser.add_argument('--cps', type=str, default=None)
args = parser.parse_args()

# Load configuration for dataset
config = Config(args.task)

if __name__ == '__main__':
    ids_list = read_list('test', task=args.task)
    results_all_folds = []
    txt_path = f"./logs/{args.exp}/evaluation_res.txt"

    print("\n Evaluating...")
    with open(txt_path, 'w') as fw:
        for fold in range(1, args.folds + 1):
            test_cls = [i for i in range(1, config.num_cls)]
            values = np.zeros((len(ids_list), len(test_cls), 2))  # Dice and ASD

            for idx, data_id in enumerate(tqdm(ids_list)):
                pred_path = os.path.join("./logs", args.exp, f"fold{fold}", f"predictions_{args.cps}", f"{data_id}.nii.gz")
                pred = read_nifti(pred_path)

                # Load Ground Truth
                if args.task == "amos":
                    label_path = os.path.join(config.base_dir, 'labelsVa', f'{data_id}.nii.gz')
                else:
                    label_path = os.path.join(config.base_dir, 'labelsTr', f'label{data_id}.nii.gz')

                label = read_nifti(label_path).astype(np.int8)

                # Check if dimensions match before interpolation
                dd, ww, hh = label.shape
                pred_shape = pred.shape

                if pred_shape != (dd, ww, hh):
                    label = torch.FloatTensor(label).unsqueeze(0).unsqueeze(0)

                    # Resizing logic to match prediction
                    if args.task == "amos":
                        resize_shape = (
                            config.patch_size[0] + config.patch_size[0] // 8,
                            config.patch_size[1] + config.patch_size[1] // 8,
                            config.patch_size[2] + config.patch_size[2] // 8
                        )
                        label = F.interpolate(label, size=resize_shape, mode='nearest')
                    else:
                        label = F.interpolate(label, size=pred_shape, mode='nearest')

                    label = label.squeeze().numpy()
                    print(f"After resizing: pred shape={pred.shape}, label shape={label.shape}")
                    print(f"Unique pred values: {np.unique(pred)}, Unique label values: {np.unique(label)}")

                # Compute Metrics
                for i in test_cls:
                    pred_i = (pred == i)
                    label_i = (label == i)

                    if pred_i.sum() > 0 and label_i.sum() > 0:
                        dice = metric.binary.dc(pred_i, label_i) * 100
                        hd95 = metric.binary.hd95(pred_i, label_i)
                    elif pred_i.sum() > 0 and label_i.sum() == 0:
                        dice, hd95 = 0, 128
                    elif pred_i.sum() == 0 and label_i.sum() > 0:
                        dice, hd95 = 0, 128
                    elif pred_i.sum() == 0 and label_i.sum() == 0:
                        dice, hd95 = 1, 0

                    values[idx][i - 1] = np.array([dice, hd95])

            # Compute Mean Values
            values_mean_cases = np.mean(values, axis=0)
            results_all_folds.append(values)

            fw.write(f"Fold {fold}\n")
            fw.write("------ Dice ------\n")
            fw.write(str(np.round(values_mean_cases[:, 0], 1)) + '\n')
            fw.write("------ ASD ------\n")
            fw.write(str(np.round(values_mean_cases[:, 1], 1)) + '\n')
            fw.write(f'Average Dice: {np.mean(values_mean_cases, axis=0)[0]}\n')
            fw.write(f'Average ASD: {np.mean(values_mean_cases, axis=0)[1]}\n')
            fw.write("=================================\n")

            print(f"Fold {fold}")
            print("------ Dice ------")
            print(np.round(values_mean_cases[:, 0], 1))
            print("------ ASD ------")
            print(np.round(values_mean_cases[:, 1], 1))
            print(np.mean(values_mean_cases, axis=0)[0], np.mean(values_mean_cases, axis=0)[1])

        # Aggregate Results Across All Folds
        results_all_folds = np.array(results_all_folds)
        results_folds_mean = results_all_folds.mean(0)

        fw.write('\n\n\nAll folds\n')
        for i in range(results_folds_mean.shape[0]):
            fw.write("=" * 5 + f" Case-{ids_list[i]}\n")
            fw.write(f'\tDice: {str(np.round(results_folds_mean[i][:, 0], 2).tolist())}\n')
            fw.write(f'\t ASD: {str(np.round(results_folds_mean[i][:, 1], 2).tolist())}\n')
            fw.write(f'\tAverage Dice: {np.mean(results_folds_mean[i], axis=0)[0]}\n')
            fw.write(f'\tAverage ASD: {np.mean(results_folds_mean[i], axis=0)[1]}\n')

        # Final Results
        fw.write("=================================\n")
        fw.write("Final Dice of each class\n")
        fw.write(str([round(x, 1) for x in results_folds_mean.mean(0)[:, 0].tolist()]) + '\n')
        fw.write("Final ASD of each class\n")
        fw.write(str([round(x, 1) for x in results_folds_mean.mean(0)[:, 1].tolist()]) + '\n')

        print("=================================")
        print("Final Dice of each class")
        print(str([round(x, 1) for x in results_folds_mean.mean(0)[:, 0].tolist()]))
        print("Final ASD of each class")
        print(str([round(x, 1) for x in results_folds_mean.mean(0)[:, 1].tolist()]))

        std_dice = np.std(results_all_folds.mean(1).mean(1)[:, 0])
        std_hd = np.std(results_all_folds.mean(1).mean(1)[:, 1])

        fw.write(f'Final Avg Dice: {round(results_folds_mean.mean(0).mean(0)[0], 2)} ± {round(std_dice, 2)}\n')
        fw.write(f'Final Avg ASD: {round(results_folds_mean.mean(0).mean(0)[1], 2)} ± {round(std_hd, 2)}\n')

        print(f'Final Avg Dice: {round(results_folds_mean.mean(0).mean(0)[0], 2)} ± {round(std_dice, 2)}')
        print(f'Final Avg ASD: {round(results_folds_mean.mean(0).mean(0)[1], 2)} ± {round(std_hd, 2)}')
