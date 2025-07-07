
# PseuAgg: Pseudo Aggregation for Decentralized Federated Learning

This repository contains the implementation of **PseuAgg**, a pseudo aggregation technique designed to reduce communication costs in decentralized federated learning (DFL). The framework supports configurable experiments across datasets, topologies, and model types.

## ⚙️ Configuration

Default configurations are specified in `cfg/config.yaml`. These include default dataset paths, model types, training parameters, and more. You can override any value programmatically in the `experiment()` function (see below).


-   `task`: "mnist", "fashion-mnist" or "cifar10".                   
-   `data_path`: same as data folder.
-   `model_type`: "simple", "conv" or "resnet18".
-   `batch_size`: interger.
-   `test_batch_size`: integer.
-   `epochs`: interger.
-   `lr`: float (lerning rate).
-   `gamma`: float.
-   `alpha_dis`: -1 or > 0, if `alpha_dis` is -1, then perform sort-and-partition distribution, else perform Dirichlet distribution with alpha = alpha_dis.
-   `client_num`: integer.
-   `device`: "cuda" or "cpu", auto switch to "cpu" if GPU is not available.
-   `seed`: interger.
-   `client_train_log`: Output log for each client (stdout).
-   `log_interval`: Output log (general info) after `log_interval` epochs.
-   `save_model`: bool (currently have not implemented).
-   `save_path`: str.
-   `topo`: "dis" (disconnected), "ring", "star", "random" or "fully".
-   `result_file`: name of result file, stored in `save_path` directory.
-   `overwrite_res`: bool (overwrite in case `result_file` is already exist).
-   `agg`: bool (perfrom aggregation step or not).
-   `agg_epoch`: integer (perform aggregation step after `agg_epoch` epochs).
-   `pseu_agg`: False (perfrom pseu-agg or not).


---

## 🚀 Running an Experiment

To launch an experiment, you can use the following pattern in `main.py` or `main.ipynb`:

```python
from experiment import experiment

def load_config(path="./cfg/config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
config = load_config()

experiment(
    name="Test",
    config=config,
    seed=0,
    task='cifar10',
    data_path='./data/cifar10/',
    topo='ring',
    epochs=10,
    alpha_dis=0.3,
    model_type='resnet18',
    client_num=10,
    visualize_data=0,
    overwrite_res=True,
    agg_epoch=2,
    result_file="avg.json"
)
```

All parameters can be overridden from the config via this function call. You can also add `visualize_data` parameter to plot the data distribution for a client.

---

## 🧪 Supported Features

- **Topologies**: Ring, Star, Fully-connected, Random, etc.
- **Datasets**: MNIST, Fashion MNIST, CIFAR-10
- **Models**: Simple FC networks, Simple Conv networks, ResNet18
- **Custom Aggregation**: Includes baseline DFedAvgM and proposed PseuAgg
- **Non-IID Control**: Via Dirichlet distribution (`alpha_dis`)
- **Visualization**: Model trajectories, accuracy curves, network topologies

---

## 📊 Results

Results are saved as `.json` files in the `data/result/` directory. You can set the file name via the `result_file` parameter.

---

## 📦 Installation

```bash
# Create environment
conda create -n pseuagg python=3.8
conda activate pseuagg

# Install requirements
pip install -r requirements.txt
```

---

## 📈 Visualization

Use `visualize.py` or `main.ipynb` to visualize model convergence, communication patterns, and final accuracy results.

---

## 📝 Citation

If you use this code, please cite the corresponding paper (to be added).

---

## 📬 Contact

For questions or contributions, feel free to open an issue or contact the author.
