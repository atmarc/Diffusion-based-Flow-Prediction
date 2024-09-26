import zipfile
import argparse

from airfoil_diffusion.airfoil_datasets import *
from airfoil_diffusion.networks import *
from airfoil_diffusion.trainer import *


def load_dataset():
    if not os.path.exists("./datasets/1_parameter/data/"):
        files=[file for file in os.listdir("./datasets/1_parameter/") if file.endswith(".zip")]
        for file in tqdm(files): 
            f=zipfile.ZipFile("./datasets/1_parameter/"+file,'r')
            for file in f.namelist():
                f.extract(file,"./datasets/1_parameter/data/")
            f.close() 

    dataset = AirfoilDataset(
        FileDataFiles(
            "./datasets/1_parameter/train_cases.txt",
            base_path="./datasets/1_parameter/data/"
        ),
        data_size=32
    )

    return dataset


def train(args):
    train_dataset = load_dataset()

    if args.arch == "unet":
        network_configs = {
        "dim_basic": 16,
        "dim_in": 3,
        "dim_multipliers": args.dim_multipliers,
        "dim_out": 3,
        "skip_connection_scale": 0.707,
        }
        network = UNet(**network_configs)
        network.show_current_configs()
            
    else:
        network_configs = {
            "attention_layers": [2, 3],
            "condition_layers": [-2],
            "depth_each_layer": 2,
            "dim_basic": 16,
            "dim_condition": 3,
            "dim_encoded_time": 8,
            "dim_in": 3,
            # "dim_multipliers": [1, 2, 4, 4],
            "dim_multipliers": args.dim_multipliers,
            "dim_out": 3,
            "heads_attention": 4,
            "linear_attention": False,
            "skip_connection_scale": 0.707,
            "use_input_condition": True
        }

        network = AifNet(**network_configs)
        network.show_current_configs()

    diffusion_trainer = DiffusionTrainer()
    train_configs = {
        "name": "training",
        "save_path": "./training/single_parameter/32/",
        "device": f"cuda:{args.device}",
        "batch_size_train": args.batch,
        "shuffle_train": True,
        "num_workers_train": 0,
        "validation_epoch_frequency": 0,
        "optimizer": "AdamW",
        "lr_scheduler": "step",
        "warmup_epoch": 0,
        "record_iteration_loss": False,
        "epochs": args.epochs,
        "save_epoch": 5000,
        "lr": args.lr,
        "final_lr": args.final_lr,
        "prune_type": args.prune_type,
        "n_prune": args.n_prune,
        "prune_perc": args.prune_perc,
        "prune_interv": args.prune_interv,
        "prune_warmup": args.prune_warmup
    }

    diffusion_trainer.train_from_scratch(network, train_dataset, **train_configs)


def parse_args():
    parser = argparse.ArgumentParser(description='CmdLine Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', default=125000, type=int, help='training epochs')
    parser.add_argument('--batch', default=25, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--final_lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--device', default=0, type=int, help='Index of GPU to use for training')    
    parser.add_argument('--arch', default="aifnet", type=str, help="Options: unet, aifnet")
    parser.add_argument('--dim_multipliers', default=[1, 2, 4, 4], type=int, nargs="+")
    parser.add_argument('--prune_type', default="L2", type=str)
    parser.add_argument('--n_prune', default=0, type=int)
    parser.add_argument('--prune_interv', default=1, type=int)
    parser.add_argument('--prune_perc', default=0.0, type=float)
    parser.add_argument('--prune_warmup', default=0, type=int, help="Number of epochs to wait until we start pruning.")
    
    args = parser.parse_args()
    params = {}
    params.update(vars(args))

    return args


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()