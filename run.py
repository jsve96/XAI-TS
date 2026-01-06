import argparse
#from tsai.all import *
from tsai.basics import *
from TSC.cfts.metrics import *
from tqdm import tqdm
import torch
from utils import *
import numpy as np
import pandas as pd



METRICS = {
    'l2': l2_distance,
    'l1': manhattan_distance,
    'fid': frechet_distance,
    'validity': prediction_change,
    'target_class_proba': class_probability_confidence,
    'spectral_similarity':spectral_similarity,
    'autocorrelation':autocorrelation_preservation,
    'sparsity':percentage_changed_points
}


#tsevo slow, wachter_genetic slow

def main(name,classifier_name,device,cf_method):
    # name = args.name
    # print(name)
    # classifier_name = args.classifier
    # device = args.device
    # cf_method = args.method

    X,y,_ = load_data(name)
    clf = load_model(classifier_name,name,device)
    print(clf)
    n_classes = int(len(set(y)))
    func_cf = execute_function(cf_method)

    metrics = {k: [] for k in METRICS}

    if cf_method == "STFT":
        cf_gen = func_cf(
                        clf,
                        lam_time=0.05,#1e-5,
                        lam_perturb=0.5,#1e-2,
                        lam_clf=5*n_classes,
                        lam_entropy = 5*n_classes,
                        steps=5000,
                        hop_length=1,
                        n_fft=int(X.shape[2] // int(X.shape[2]/4)),
                        device=device,
                        init_scale=1e-3,
                        lr = 0.001
                    )
        BATCH_SIZE=64
        metrics = generate_STFT(X,y,METRICS,cf_gen,BATCH_SIZE)
    else:
        print('hello')

        #metrics = generate_cf(X,METRICS,func_cf,clf.model,device=device)
        BATCH_SIZE=64
        metrics = generate_cf_batch(X,METRICS,func_cf,clf.model,device,BATCH_SIZE)



    

    # for _,v in metrics.items():
    #     print(_,np.mean(v))
    print(pd.DataFrame(metrics).mean())
    return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='name of dataset',type=str, default='ECG200')
    parser.add_argument('--classifier', help='name of classifier',type=str,default='HydraPlus')
    parser.add_argument('--device',type=str,default='cuda')
    parser.add_argument('--method',type=str,default='STFT')
    args = parser.parse_args()

    name = args.name
    classifier_name = args.classifier
    device = args.device
    cf_method = args.method

    main(name,classifier_name,device,cf_method)