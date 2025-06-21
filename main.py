import argparse
import yaml
from experiment import experiment

def load_config(path="./cfg/config.yaml"):
    # pylint: disable=unspecified-encoding
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Override config from YAML")

    # Add overrides for all fields in YAML
    parser.add_argument('--name', type=str, default="DFL experiment")
    parser.add_argument('--task', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--alpha_dis', type=float)
    parser.add_argument('--client_num', type=int)
    parser.add_argument('--device', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--client_train_log', action='store_true')
    parser.add_argument('--log_interval', type=int)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--topo', type=str)
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--overwrite_res', action='store_true')
    parser.add_argument('--agg', action='store_true')
    parser.add_argument('--agg_epoch', type=int)
    parser.add_argument('--pseu_agg', action='store_true')

    return parser.parse_args()

def override_config(config_, args_):
    args_dict = vars(args_)
    for key, value in args_dict.items():
        # Override only if explicitly passed in CLI
        if value is not None:
            config_[key] = value
        # Handle bool flags properly: if not passed, keep YAML; if passed, set to True
        elif isinstance(config_.get(key), bool) and getattr(args_, key) is True:
            config_[key] = True
    return config_

if __name__ == "__main__":
    config = load_config()
    args = parse_args()
    config = override_config(config, args)
    experiment(args.name, config)
