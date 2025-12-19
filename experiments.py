# launcher.py
import argparse
from run import main
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "configs"

RESULTS_DIR = BASE_DIR/"results"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--set',
        default='01',
        help='List of datasets to run',
        type=str
    )
    parser.add_argument('--classifier', type=str, default='HydraPlus')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--method', type=str, default='STFT')


    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()


    #load datasets

    with open(f"{str(CONFIG_DIR)}/SETS/SET-{args.set[0]}/Set-{args.set[0]}{args.set[1]}.json","rb") as f:
        datasets = json.load(f)

    print(datasets)

    key = list(datasets.keys())
    print(key)
    _METRICS_ = {}
    for dataset in datasets[key[0]]:
        print(dataset)

        print(f"\n=== Running dataset: {dataset} ===")

        name = dataset
        classifier_name = args.classifier
        device = args.device
        cf_method = args.method
        _METRICS_[dataset] = main(name,classifier_name,device,cf_method)
## NEED TO SAVE METRICS --> construc FOLDER for SETS either classifier/
    print(_METRICS_)
