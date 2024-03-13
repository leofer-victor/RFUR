import os

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Logger:
    def __init__(self, scheduler):
        self.proj_path = os.path.dirname(os.path.dirname(__file__))
        self.scheduler = scheduler
        self.writer = None
        now = datetime.now()
        self.now_str = now.strftime('%Y-%m-%d_%H:%M:%S')
        self.writer = SummaryWriter(os.path.join(self.proj_path, 'data', 'logs'))

    def print_training_status(self, epoch, epochs, best_epoch, train_epoch_loss, val_epoch_loss, lowest_loss):
        print('\n epoch: {}/{}, best_epoch: {}, training_loss: {:.6f}, val_loss: {:.6f}, lowest_loss: {:.6f}, scheduler: {:.8f}'.format(
            epoch + 1, epochs, best_epoch, train_epoch_loss, val_epoch_loss, lowest_loss, self.scheduler.get_last_lr()[0]))

    def write(self, train_epoch_loss, val_epoch_loss, epoch):
        self.writer.add_scalar('train_loss', train_epoch_loss, epoch)
        self.writer.add_scalar('val_loss', val_epoch_loss, epoch)

    def close(self):
        self.writer.close()

    def update_record(self, best_epoch, current_epoch, lowest_loss):
        readFile = open(self.proj_path + '/data/experiment_diary/{}.txt'.format(self.now_str))
        lines = readFile.readlines()
        readFile.close()

        file = open(self.proj_path + '/data/experiment_diary/{}.txt'.format(self.now_str), 'w')
        file.writelines([item for item in lines[:-2]])
        file.write('Best_epoch: {}/{}\n'.format(best_epoch, current_epoch))
        file.write('Val_loss: {:.4f}'.format(lowest_loss))
        file.close()

    def record(self, args):
        file = open(self.proj_path + '/data/experiment_diary/{}.txt'.format(self.now_str), 'a+')
        file.write('Time_str: {}\n'.format(self.now_str))
        file.write('Model_name: {}\n'.format(args.name))
        file.write('Device_no: {}\n'.format(args.gpus))
        file.write('Epochs: {}\n'.format(args.epochs))
        file.write('Learning_rate: {}\n'.format(args.learning_rate))
        file.write('Neighbour_slices: {}\n'.format(args.neighbour_slice))
        file.write('Best_epoch: 0\n')
        file.write('Val_loss: {:.4f}\n'.format(1000))
        file.close()
        print('Information has been saved!')