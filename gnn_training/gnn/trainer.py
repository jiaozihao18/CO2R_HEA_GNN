import torch
import torch.nn as nn
import shutil
import logging
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


class Trainer:
    """
    Trainer for a neural network model.

    Parameters
    ----------
    model : nn.Module
        The neural network model to be trained.
    train_loader : DataLoader
        DataLoader for the training data.
    val_loader : DataLoader
        DataLoader for the validation data.
    test_loader : DataLoader
        DataLoader for the test data.
    kwargs : dict
        Additional configuration parameters.

        optimizer_type : str, optional (default='Adam')
            The type of optimizer to use (e.g., 'Adam', 'SGD').
        optimizer_params : dict, optional (default={})
            Parameters for the optimizer.
        lr_scheduler_type : str, optional (default=None)
            The type of learning rate scheduler to use (e.g., 'StepLR').
        lr_scheduler_params : dict, optional (default={})
            Parameters for the learning rate scheduler.
        max_epoch : int, optional (default=40)
            Maximum number of training epochs.
        print_freq : int, optional (default=10)
            Frequency of printing training status.
        checkpoint_path : str, optional (default='checkpoint.pth.tar')
            Path to save the checkpoint.
        best_model_path : str, optional (default='model_best.pth.tar')
            Path to save the best model.
        early_stopping_patience : int, optional (default=300)
            Number of epochs to wait for improvement before early stopping.

    """
    def __init__(self, 
                 model, 
                 train_loader, 
                 val_loader, 
                 test_loader,
                 config,
                 resume = False) -> None:
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.resume = resume
        self.config = config      

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

        # Dynamically create the optimizer
        optimizer_type = config['config_trainer'].get("optimizer_type", "Adam")  # Get the optimizer type
        optimizer_params = config['config_trainer'].get("optimizer_params", {})  # Get the optimizer parameters
        optimizer_class = getattr(optim, optimizer_type, None)
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)

        # Dynamically create the learning rate scheduler
        lr_scheduler_type = config['config_trainer'].get("lr_scheduler_type", None)
        lr_scheduler_params = config['config_trainer'].get("lr_scheduler_params", {})
        if lr_scheduler_type:
            lr_scheduler_class = getattr(lr_scheduler, lr_scheduler_type, None)
            if lr_scheduler_class is None:
                raise ValueError(f"Unsupported lr scheduler type: {lr_scheduler_type}")
            self.lr_scheduler = lr_scheduler_class(self.optimizer, **lr_scheduler_params)
        else:
            self.lr_scheduler = None

        self.best_mae_error = 1e10

        self.start_epoch = 0
        self.max_epoch = config['config_trainer'].get("max_epoch", 40)
        self.print_freq = config['config_trainer'].get("print_freq", 10)
        
        self.checkpoint_path = config['config_trainer'].get("checkpoint_path", "checkpoint.pth.tar")
        self.best_model_path = config['config_trainer'].get("best_model_path", "model_best.pth.tar")

        self.early_stopping_patience = config['config_trainer'].get("early_stopping_patience", 300)
        self.early_stopping_counter = 0

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        
        if self.resume:
            if os.path.isfile(self.resume):
                self.load_checkpoint(filename=self.resume)
            else:
                logging.info("=> no checkpoint found at '{}'".format(self.resume))
    
    def load_checkpoint(self, filename='checkpoint.pth.tar'):
        logging.info("=> loading checkpoint %s"%filename)
        checkpoint = torch.load(filename)
        self.start_epoch = checkpoint['epoch']
        self.best_mae_error = checkpoint['best_mae_error']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logging.info("=> loaded checkpoint %s (epoch %s)"%(filename, checkpoint['epoch']))

    def run(self):
        """
        Main training loop. Trains the model and validates it after each epoch.
        Saves the best model based on validation MAE error.
        """
        
        if self.resume:
            for epoch in range(0, self.start_epoch):
                pass
            
        for epoch in range(self.start_epoch, self.max_epoch):
            self.epoch = epoch

            self.train()
            mse_error, mae_error = self.validate()

            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(mae_error)  # ReduceLROnPlateau needs the validation loss
                else:
                    self.lr_scheduler.step()  # Other schedulers do not need the validation loss

            # Remember the best MAE error and save checkpoint
            is_best = mae_error < self.best_mae_error
            self.best_mae_error = min(mae_error, self.best_mae_error)

            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_mae_error': self.best_mae_error,
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                'config': self.config
            }, is_best)

            # Early stopping
            if is_best:
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping_patience:
                    logging.info("Early stopping")
                    break

        # Test the best model
        best_checkpoint = torch.load(self.best_model_path)
        self.model.load_state_dict(best_checkpoint['state_dict'])
        self.validate(test=True)

    def train(self):
        """
        Training loop for one epoch.
        """
        self.model.train()

        mse_avg = AverageMeter()
        mae_avg = AverageMeter()
        time_batch_gen = AverageMeter()
        time_batch_proc = AverageMeter()

        time_start = time.time()
        for i, batch in enumerate(self.train_loader):
            time_batch_gen.update(time.time()-time_start)
            batch = to_device(batch, self.device)

            # Forward pass
            output = self.model(batch).squeeze(-1)

            # Calculate losses
            mse_loss = self.mse_loss(output, batch['f_energy'])
            mae_loss = self.mae_loss(output, batch['f_energy'])

            # Update loss meters
            mse_avg.update(mse_loss.item(), batch['f_energy'].size(0))
            mae_avg.update(mae_loss.item(), batch['f_energy'].size(0))

            # Backward pass and optimization
            self.optimizer.zero_grad()
            mse_loss.backward()
            self.optimizer.step()
            
            # Get the current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            time_batch_proc.update(time.time()-time_start)
            
            # Print training status
            if i % self.print_freq == 0:
                logging.info("Epoch: %d, Step [%d/%d], Current MSE Loss: %.4f, Current MAE Loss: %.4f, "
                             "Avg MSE Loss: %.4f, Avg MAE Loss: %.4f, LR: %.6f, "
                             "Time gen val-avg: %.3f-%.3f, "
                             "Time proc val-avg: %.3f-%.3f" % 
                             (self.epoch, i, len(self.train_loader), mse_avg.val, mae_avg.val, 
                              mse_avg.avg, mae_avg.avg, current_lr, 
                              time_batch_gen.val, time_batch_gen.avg, 
                              time_batch_proc.val, time_batch_proc.avg))

            time_start = time.time()

    def validate(self, test=False):
        """
        Validation loop.
        """
        self.model.eval()

        mse_avg = AverageMeter()
        mae_avg = AverageMeter()

        data_loader = self.test_loader if test else self.val_loader

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                batch = to_device(batch, self.device)

                output = self.model(batch).squeeze(-1)

                mse_loss = self.mse_loss(output, batch['f_energy'])
                mae_loss = self.mae_loss(output, batch['f_energy'])

                mse_avg.update(mse_loss.item(), batch['f_energy'].size(0))
                mae_avg.update(mae_loss.item(), batch['f_energy'].size(0))

        dataset_type = "Test" if test else "Val"
        logging.info(f"** {dataset_type} dataset ** Epoch: {self.epoch}, Avg MSE Loss: {mse_avg.avg:.4f}, Avg MAE Loss: {mae_avg.avg:.4f}")
        return mse_avg.avg, mae_avg.avg

    def save_checkpoint(self, state, is_best, filename=None):
        """
        Save the model checkpoint.
        """
        filename = filename or self.checkpoint_path
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, self.best_model_path)
