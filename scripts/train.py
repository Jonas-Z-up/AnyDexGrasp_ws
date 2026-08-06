""" Training routine for GSNet baseline model. """

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummayWriter
torch.multiprocessing.set_sharing_strategy('file_system')

import MinkowskiEngine as ME

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from graspnet import GraspNetVoxelizationDataset, collate_fn, load_grasp_labels, convert_data_to_gpu
from loss import GraspLoss
from solvers import PolyLR

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='minkowski_graspnet_single_point', help='Model file name [default: minkowski_graspnet_single_point]')
parser.add_argument('--dataset', default='graspnet', help='Dataset name. ycb or graspnet. [default: graspnet]')
parser.add_argument('--camera', default='kinect', help='data source [realsense/kinect] [default: kinect]')
parser.add_argument('--stage', default='full', help='Training stage (stage1/stage2/full). [default: stage1]')
parser.add_argument('--stage1_checkpoint_path', default=None, help='Model stage 1 checkpoint path [default: None]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='logs/model/rep_log', help='Dump dir to save model checkpoint')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--max_epoch', type=int, default=18, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.1, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=2, help='Period of BN decay (in epochs) [default: 10]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='8,12,16', help='When to decay the learning rate (in epochs) [default: 40,60,80]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--centralize_points', action='store_true', help='Point centralization.')
parser.add_argument('--half_views', action='store_true', help='Use only half views in network.')
FLAGS = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
DATASET = FLAGS.dataset
CAMERA = FLAGS.camera
BATCH_SIZE = FLAGS.batch_size
NUM_VIEW = FLAGS.num_view
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
LOG_DIR = FLAGS.log_dir
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
STAGE1_CHECKPOINT_PATH = FLAGS.stage1_checkpoint_path
STAGE = FLAGS.stage
assert(STAGE == 'stage1' or STAGE == 'stage2' or STAGE == 'full')
if STAGE == 'stage2':
    assert(STAGE1_CHECKPOINT_PATH is not None)
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH

# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(LOG_DIR) and FLAGS.overwrite:
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)'%(LOG_DIR))
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')
        os.system('rm -r %s'%(LOG_DIR))

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

# Create Dataset and Dataloader
graspnet_v1_root = 'logs/data/representation_model/graspnet_v1_newformat'
valid_obj_idxs, grasp_labels = load_grasp_labels(graspnet_v1_root)
TRAIN_DATASET = GraspNetVoxelizationDataset(graspnet_v1_root, valid_obj_idxs, grasp_labels, camera=CAMERA, split='train', remove_outlier=True, remove_invisible=True, augment=True, heatmap='scene', score_as_heatmap=False, score_as_view_heatmap=False, heatmap_th=0.6, view_heatmap_th=0.6, centralize_points=FLAGS.centralize_points)
TEST_DATASET = GraspNetVoxelizationDataset(graspnet_v1_root, valid_obj_idxs, grasp_labels, camera=CAMERA, split='test_seen', remove_outlier=True, remove_invisible=True, augment=False, centralize_points=FLAGS.centralize_points)

print(len(TRAIN_DATASET), len(TEST_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=5, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=5, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))
# Init the model and optimzier
MODEL = importlib.import_module(FLAGS.model) # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = GraspLoss()

net = MODEL.MinkowskiGraspNet(num_depth=5, half_views=FLAGS.half_views)
if torch.cuda.device_count() > 1:
    log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
net.to(device)
# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)
# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))

lr_scheduler = PolyLR(optimizer, max_iter=MAX_EPOCH, power=0.9, last_step=start_epoch-1)
EPOCH_CNT = 0

# TFBoard Visualizers
TRAIN_WRITER = SummaryWriter(os.path.join(LOG_DIR, 'train'))
TEST_WRITER = SummaryWriter(os.path.join(LOG_DIR, 'test'))

# ------------------------------------------------------------------------- GLOBAL CONFIG END

def train_one_epoch():
    stat_dict = {} # collect statistics
    # adjust_learning_rate(optimizer, EPOCH_CNT)
    # bnm_scheduler.step() # decay BN momentum
    # set model to training mode
    net.train()
    data_time = 0.
    net_time = 0.
    tic = time.time()
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        # Process input
        # Upd Note. "SparseTensor.to()" is deleted in ME v0.5.
        batch_data_label = convert_data_to_gpu(batch_data_label)
        batch_data_label['sinput'] = ME.SparseTensor(batch_data_label['feats'], batch_data_label['coords'])
        
        toc = time.time()
        data_time += toc - tic

        tic = time.time()
        # Forward pass
        end_points = net(batch_data_label)

        # Compute loss and gradients, update parameters.
        loss, end_points = criterion(end_points)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or '_acc' in key or '_ratio' in key or '_prec' in key or '_recall' in key or '_count' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()
            if '_dist' in key:
                if (batch_idx+1) % 100 == 0:
                    stat_dict[key] = end_points[key].cpu().detach().numpy()
        toc = time.time()
        net_time += toc - tic

        batch_interval = 10
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            log_string('data time: %fs' % (data_time/batch_interval))
            log_string('net  time: %fs' % (net_time/batch_interval))
            for key in sorted(stat_dict.keys()):
                if '_dist' in key:
                    if (batch_idx+1) % 100 == 0:
                        TRAIN_WRITER.add_histogram(key, stat_dict[key], (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*BATCH_SIZE)
                    continue
                else:
                    TRAIN_WRITER.add_scalar(key, stat_dict[key]/batch_interval, (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*BATCH_SIZE)
                log_string('mean %s: %f'%(key, stat_dict[key]/batch_interval))
                stat_dict[key] = 0
            data_time = 0.
            net_time = 0.

        tic = time.time()

def evaluate_one_epoch():
    stat_dict = {} # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d'%(batch_idx))
        # Upd Note. "SparseTensor.to()" is deleted in ME v0.5.
        batch_data_label = convert_data_to_gpu(batch_data_label)
        batch_data_label['sinput'] = ME.SparseTensor(batch_data_label['feats'], batch_data_label['coords'])
        
        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data_label)

        # Compute loss
        with torch.no_grad():
            loss, end_points = criterion(end_points)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or '_acc' in key or '_ratio' in key or '_prec' in key or '_recall' in key or '_count' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()
            if '_dist' in key:
                if (batch_idx+1) % 100 == 0:
                    stat_dict[key] = end_points[key].cpu().detach().numpy()

    # Log statistics
    for key in sorted(stat_dict.keys()):
        if '_dist' in key:
            if (batch_idx+1) % 100 == 0:
                TEST_WRITER.add_histogram(key, stat_dict[key], (EPOCH_CNT+1)*len(TRAIN_DATALOADER)*BATCH_SIZE)
            continue
        else:
            TEST_WRITER.add_scalar(key, stat_dict[key]/float(batch_idx+1), (EPOCH_CNT+1)*len(TRAIN_DATALOADER)*BATCH_SIZE)
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    mean_loss = stat_dict['losses/overall']/float(batch_idx+1)
    return mean_loss


def train(start_epoch):
    global EPOCH_CNT 
    min_loss = 1e10
    loss = 0
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f'%(lr_scheduler.get_last_lr()[0]))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        # Train
        np.random.seed()
        train_one_epoch()
        lr_scheduler.step()
        # Save checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))
        # Eval
        if epoch % 3 == 0:
            loss = evaluate_one_epoch()

if __name__=='__main__':
    train(start_epoch)
