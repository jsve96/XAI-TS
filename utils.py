import argparse
from tsai.all import load_learner, TSClassifier, TSClassification, TSStandardize, accuracy, get_classification_data
from tsai.basics import *
from TSC.cfts.metrics import *
import importlib
from tqdm import tqdm
import torch
import numpy as np

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"





def load_data(name):
    X, y, splits = get_classification_data(name, split_data=False)
    X_norm = (np.array(X.tolist())[:,0,:]-np.array(X.tolist())[:,0,:].mean(axis=0))/(np.array(X.tolist())[:,0,:].std(axis=0))
    X= X_norm[:,np.newaxis,:]
    return X, y, splits



# def load_model(classifier,dataname,device):
#     model_path = MODEL_DIR / f"{classifier}-{dataname}.pkl"
#     print(f"try loading model from {model_path}")
#     #print(type(model_path))
#     try:
#         module_name = f"tsai.models.{classifier}"
#         tsai_module = importlib.import_module(module_name)

#         #model_path = MODEL_DIR / f"{classifier}-{dataname}.pkl"
#         clf = load_learner(model_path,cpu=False)
#         #clf = torch.load(model_path, map_location='cuda', weights_only=True)

#     except:
#         print(f"{classifier} not found in /models ")
#         print("Initialize training")
#         X,y,splits = load_data(dataname)
#         tfms = [None, TSClassification()]
#         batch_tfms = TSStandardize()
#         clf = TSClassifier(X, y, splits=splits, path=MODEL_DIR, arch=classifier, tfms=tfms, batch_tfms=batch_tfms, metrics=accuracy,device=device)
#         clf.fit_one_cycle(100, 3e-4)
#         clf.export(f"{classifier}-{dataname}.pkl")

#         return clf

def load_model(classifier: str, dataname: str, device: str):
    model_path = MODEL_DIR / f"{classifier}-{dataname}.pkl"
    print(f"Trying to load model from {model_path}")

    try:
        # Dynamically import the model class from tsai
        module_name = f"tsai.models.{classifier}"
        tsai_module = importlib.import_module(module_name)
        model_class = getattr(tsai_module, classifier)  # get the actual class

        # Load the learner (class must be imported first!)
        clf = load_learner(model_path, cpu=(device=='cpu'))
        return clf

    except Exception as e:
        print(f"Model not found or failed to load: {e}")
        print("Initializing training...")

        # Dynamically import the model class if not already
        if 'model_class' not in locals():
            module_name = f"tsai.models.{classifier}"
            tsai_module = importlib.import_module(module_name)
            model_class = getattr(tsai_module, classifier)

        # Load data
        X, y, splits = load_data(dataname)

        # Create and train TSClassifier
        tfms = [None, TSClassification()]
        batch_tfms = TSStandardize()
        clf = TSClassifier(
            X, y,
            splits=splits,
            path=MODEL_DIR,
            arch=model_class,  # pass the class, not string
            tfms=tfms,
            batch_tfms=batch_tfms,
            metrics=accuracy,
            device=device,
        )

        clf.fit_one_cycle(100, 3e-4)
        clf.export(model_path.name)

        return clf


def execute_function(method):
    module_name = f"TSC.cfts.cf_{method}.{method}"
    if method == 'STFT':
        train_module = importlib.import_module(module_name)
        func = 'TSCounterfactualGenerator'
        cf_function = getattr(train_module, func)
        print(f"{module_name} loaded")
        return cf_function
    else:
        try:
            train_module = importlib.import_module(module_name)
            func = 'cf_ts'
            cf_function = getattr(train_module, func)
            print(f"{module_name} loaded")
        except ModuleNotFoundError:
            print(f"Module {module_name} not found.")
            exit(1)
        except AttributeError:
            print(f"Function {func} not found in module {module_name}.")
            exit(1)
        return cf_function



def encode_labels(y):
    """
    y: list / np.ndarray / torch.Tensor of strings or numbers
    returns:
        y_encoded: torch.LongTensor (B,)
        class_to_idx: dict mapping original label -> int
    """
    # Convert to numpy strings
    y_np = np.asarray(y).astype(str)

    # Unique sorted labels (stable across runs)
    classes = np.unique(y_np)

    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    y_encoded = np.vectorize(class_to_idx.get)(y_np)

    return torch.tensor(y_encoded, dtype=torch.long), class_to_idx


def generate_STFT(X,y,METRICS,cf_gen,BATCH_SIZE):
    N = X.shape[0]
    metrics = {k: [] for k in METRICS}
    device = cf_gen.device
    clf = cf_gen.learner
    y,_ = encode_labels(y)
    for start in tqdm(range(0, N, BATCH_SIZE)):
            end = min(start + BATCH_SIZE, N)

            # ------------------------------------------------
            # Batch inputs
            # ------------------------------------------------
            x = torch.tensor(X[start:end]).to(device).float()  # (B, C, T)
            y_true = torch.tensor(y[start:end]).to(device).float()

            # ------------------------------------------------
            # Compute batch targets (2nd most probable class)
            # ------------------------------------------------
            with torch.no_grad():
                probas = torch.softmax(clf.model(x), dim=-1)
                target = torch.topk(probas, k=2, dim=-1).indices[:, 1]  # (B,)
                y_clean_pred = probas.argmax(dim=1)
            
            active = y_clean_pred ==  y_true      # (B,)
            active_idx = active.nonzero(as_tuple=True)[0]



            # ------------------------------------------------
            # Generate counterfactuals (batched!)
            # ------------------------------------------------
            cf = cf_gen.generate(x[active_idx], target[active_idx])
            x_cf = cf["x_cf"]
            print(x[active_idx].shape)
            # ------------------------------------------------
            # Metric computation (per sample)
            # ------------------------------------------------
            x_np = x[active_idx].detach().cpu().numpy()
            x_cf_np = x_cf.detach().cpu().numpy()

            for i in range(x_np.shape[0]):
                xi = x_np[i:i+1]
                xcf_i = x_cf_np[i:i+1]
                ti = target[active_idx][i].detach()

                for k, f_dist in METRICS.items():
                    if k in ['l2', 'l1', 'fid']:
                        metrics[k].append(f_dist(xi, xcf_i))

                    elif k == 'validity':
                        metrics[k].append(
                            prediction_change(
                                torch.tensor(xi).to(device),
                                torch.tensor(xcf_i).to(device),
                                clf.model,
                                ti
                            )
                        )

                    elif k == 'target_class_proba':
                        metrics[k].append(
                            class_probability_confidence(
                                torch.tensor(xcf_i).to(device),
                                clf.model,
                                ti
                            )
                        )

                    elif k == 'spectral_similarity':
                        metrics[k].append(spectral_similarity(xi,xcf_i))
                    
                    elif k == "autocorrelation":
                        metrics[k].append(autocorrelation_preservation(xi,xcf_i))
                    
                    elif k == "sparsity":
                        metrics[k].append(percentage_changed_points(xi,xcf_i))

    return metrics



def generate_cf(X,METRICS,func_cf,model,device):
    metrics = {k: [] for k in METRICS}

    for x in tqdm(X):
        #print(torch.tensor(x).double())
        probas = torch.softmax(model(torch.tensor(x).float().to(device).reshape(1,1,-1)), dim=-1)
        target = torch.topk(probas, k=2, dim=-1).indices[:, 1]  # (B,
        cf, prob = func_cf(x,dataset=X,model=model) # (1,L) shape
        
        if cf is None:
             #print(cf.shape)
            for k, f_dist in METRICS.items():
                  metrics[k].append(np.nan)
        else:
            if cf.shape[1] == 1:
                cf = cf.reshape(1,-1)
            for k, f_dist in METRICS.items():
                            if k in ['l2', 'l1', 'fid']:
                                metrics[k].append(f_dist(x, cf))

                            elif k == 'validity':
                                metrics[k].append(
                                    prediction_change(
                                        torch.tensor(x).float().reshape(1,1,-1).to(device),
                                        torch.tensor(cf).float().reshape(1,1,-1).to(device),
                                        model,
                                        target
                                    )
                                )

                            elif k == 'target_class_proba':
                                metrics[k].append(
                                    class_probability_confidence(
                                        torch.tensor(cf).float().reshape(1,1,-1).to(device),
                                        model,
                                        target
                                    )
                                    
                                )

                            elif k == 'spectral_similarity':
                                metrics[k].append(spectral_similarity(x,cf))

                            elif k == "autocorrelation":
                                metrics[k].append(autocorrelation_preservation(x,cf))

                            elif k == "sparsity":
                                metrics[k].append(percentage_changed_points(x,cf))




    return metrics


def generate_cf_batch(X,METRICS,func_cf,model,device,BATCH_SIZE):
    N = X.shape[0]
    metrics = {k: [] for k in METRICS}
    #clf = cf_gen.learner

    for start in tqdm(range(0, N, BATCH_SIZE)):
            end = min(start + BATCH_SIZE, N)

            # ------------------------------------------------
            # Batch inputs
            # ------------------------------------------------
            x = torch.tensor(X[start:end]).to(device).float()  # (B, C, T)
           # print(x)

            # ------------------------------------------------
            # Compute batch targets (2nd most probable class)
            # ------------------------------------------------
            with torch.no_grad():
                probas = torch.softmax(model(x), dim=-1)
                target = torch.topk(probas, k=2, dim=-1).indices[:, 1]  # (B,)

            # ------------------------------------------------
            # Generate counterfactuals (batched!)
            # ------------------------------------------------
            #cf = cf_gen.generate(x, target)
            x_cf, prob = func_cf(x,dataset=X,model=model) # (1,L) shape
            #x_cf = cf["x_cf"]
            print(x_cf.shape)
            # ------------------------------------------------
            # Metric computation (per sample)
            # ------------------------------------------------
            x_np = x.detach().cpu().numpy()
            x_cf_np = x_cf.detach().cpu().numpy()


            for i in range(x_np.shape[0]):
                xi = x_np[i:i+1]
                xcf_i = x_cf_np[i:i+1]
                ti = target[i].detach()

                for k, f_dist in METRICS.items():
                    if k in ['l2', 'l1', 'fid']:
                        metrics[k].append(f_dist(xi, xcf_i))

                    elif k == 'validity':
                        metrics[k].append(
                            prediction_change(
                                torch.tensor(xi).reshape(1,1,-1).to(device),
                                torch.tensor(xcf_i).reshape(1,1,-1).to(device),
                                model,
                                ti
                            )
                        )

                    elif k == 'target_class_proba':
                        metrics[k].append(
                            class_probability_confidence(
                                torch.tensor(xcf_i).reshape(1,1,-1).to(device),
                                model,
                                ti
                            )
                        )

                    elif k == 'spectral_similarity':
                        metrics[k].append(spectral_similarity(xi,xcf_i))
                    
                    elif k == "autocorrelation":
                        metrics[k].append(autocorrelation_preservation(xi,xcf_i))
                    
                    elif k == "sparsity":
                        metrics[k].append(percentage_changed_points(xi,xcf_i))


    return metrics
