import sys
sys.path.append("gnn")
import logging
import json
from gnn.trainer import Trainer
from gnn.model import CGCNN
from ase.db import connect
from gnn.data import get_data_loader, split_dataset_k_fold, LmdbDataset

config = {
    
    'config_data': {
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'seed': 42,
        'batch_size': 30
    },
    
    # Uncomment the following lines if you want to use test_ratio and k instead
    # 'config_data': {
    #     'test_ratio': 0.1,
    #     'k': 5,
    #     'seed': 42,
    #     'batch_size': 30
    # },
    
    'config_model': {
        'atom_fea_len': 256,
        'n_conv': 3,
        'fc_fea_len': 128,
        'n_fc': 2,
        'num_gaussians': 50,
        'cutoff': 6,
    },
    
    'config_trainer': {
        'max_epoch': 160,
        'optimizer_type': 'AdamW',
        'optimizer_params': {
            'lr': 0.005,
        },
        'lr_scheduler_type': 'ReduceLROnPlateau',
        'lr_scheduler_params': {
            'mode': 'min',
            'factor': 0.8,
            'patience': 3,
        }
    }
}



# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Print the input config parameters
logging.info(json.dumps(config, indent=4))

dataset = LmdbDataset("/home/zjia754/0hea/gnn/db2lmdb/com.lmdb")

idxs_dict = {}
for data in dataset:
    l = idxs_dict.setdefault(data.ads_name, [])
    l.append(data.sid)

train_loader, val_loader, test_loader = get_data_loader(dataset, idxs_dict,
                                                        config['config_data']['train_ratio'],
                                                        config['config_data']['val_ratio'],
                                                        config['config_data']['seed'],
                                                        config['config_data']['batch_size'])
model = CGCNN(**config['config_model'])
trainer = Trainer(model, train_loader, val_loader, test_loader,
                  resume=False,
                  config=config)
trainer.run()


# dataloaders = split_dataset_k_fold(dataset,
#                                    idxs_dict,
#                                    config_data['test_ratio'],
#                                    config_data['k'],
#                                    config_data['seed'],
#                                    config_data['batch_size'])

# for i, (train_loader, val_loader, test_loader) in enumerate(dataloaders):
#     logging.info("\n***** %s *****\n"%i)
#     model = CGCNN(**config_model)
#     trainer = Trainer(model, train_loader, val_loader, test_loader, **config_trainer)
#     trainer.run()
