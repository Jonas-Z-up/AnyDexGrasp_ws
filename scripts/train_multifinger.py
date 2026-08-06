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

torch.multiprocessing.set_sharing_strategy('file_system')

import MinkowskiEngine as ME

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from multifinger_hand import MultifingerDataset, collate_fn, convert_data_to_gpu
from solvers import PolyLR, StepLR
from loss import MultifingerType1Loss
from param import *
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default = './', help = 'dataset root')
parser.add_argument('--model', default='minkowski_graspnet', help='Model file name [default: minkowski_graspnet]')
parser.add_argument('--log_dir', default='log/Inspire/pose0-11/final', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--gripper_type', default='Allegro', help='gripper_type')
parser.add_argument('--train_multifinger_type', type=int, default=1, help='multifinger type for training')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='Optimization L2 weight decay [default: 0]')
# parser.add_argument('--bn_decay_step', type=int, default=2, help='Period of BN decay (in epochs) [default: 10]')
# parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
# parser.add_argument('--lr_decay_steps', default='20,40,60', help='When to decay the learning rate (in epochs) [default: 40,60,80]')
# parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')

FLAGS = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
DATASET_ROOT_TRAIN = os.path.join(FLAGS.dataset_root)
DATASET_ROOT_TEST = os.path.join(FLAGS.dataset_root)
NUM_MULTIFINGER_TYPE = 1
NUM_MULTIFINGER_DEPTH = 4
NUM_TWO_FINGER_ANGLE = 12
NUM_TWO_FINGER_DEPTH = 5
BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GRIPPER_TYPE = FLAGS.gripper_type
MULTIFINGER_TYPE = FLAGS.train_multifinger_type
WEIGHT_DECAY = FLAGS.weight_decay

LOG_DIR = FLAGS.log_dir


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
TRAIN_DATASET = MultifingerDataset(root = DATASET_ROOT_TRAIN, multifinger_type = GRIPPER_TYPE, 
                                   dataset_type = "train", train_type = MULTIFINGER_TYPE, num_multifinger_type = NUM_MULTIFINGER_TYPE)
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=16, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
TEST_DATASET = MultifingerDataset(root = DATASET_ROOT_TEST, multifinger_type = GRIPPER_TYPE, 
                                  dataset_type = "test", train_type = MULTIFINGER_TYPE, num_multifinger_type = NUM_MULTIFINGER_TYPE)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=16, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
log_string("Train dataset length:{}".format(len(TRAIN_DATASET)))
log_string("Test dataset length:{}".format(len(TEST_DATASET)))
# Init the model and optimzier
MODEL = importlib.import_module(FLAGS.model) # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log_string("Use Device:{}".format(device))
multifinger_net = MODEL.MinkowskiGraspNetMultifingerType1(num_multifinger_type = NUM_MULTIFINGER_TYPE, 
                                                      num_multifinger_depth = NUM_MULTIFINGER_DEPTH,
                                                      num_two_finger_angle = NUM_TWO_FINGER_ANGLE,
                                                      num_two_finger_depth = NUM_TWO_FINGER_DEPTH)
multifinger_net.to(device)

criterion = MultifingerType1Loss(num_multifinger_type = NUM_MULTIFINGER_TYPE, 
                                 num_multifinger_depth = NUM_MULTIFINGER_DEPTH,
                                 num_two_finger_angle = NUM_TWO_FINGER_ANGLE,
                                 num_two_finger_depth = NUM_TWO_FINGER_DEPTH,
                                 train_type = MULTIFINGER_TYPE)
criterion.to(device)
# Load the Adam optimizer
optimizer = optim.Adam(multifinger_net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0

lr_scheduler = PolyLR(optimizer, max_iter=MAX_EPOCH, power=0.99, last_step=start_epoch-1)
# lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=start_epoch-1)
EPOCH_CNT = 0

# TFBoard Visualizers
TRAIN_WRITER = SummaryWriter(os.path.join(LOG_DIR, 'train'))
TEST_WRITER = SummaryWriter(os.path.join(LOG_DIR, 'test'))
# ------------------------------------------------------------------------- GLOBAL CONFIG END

def save_model(epoch, mean_indicators, best_indicators):
    if (mean_indicators[0] - best_indicators[0] > 0.01 and mean_indicators[1] > 0.05) or (abs(mean_indicators[0] - best_indicators[0]) < 0.01 and mean_indicators[1] > best_indicators[1]):
        best_indicators[:3] = mean_indicators[:3].tolist()
        if not os.path.exists(os.path.join(LOG_DIR, str(MULTIFINGER_TYPE), '0.5')):
            os.makedirs(os.path.join(LOG_DIR, str(MULTIFINGER_TYPE), '0.5'))
        model_path = os.path.join(LOG_DIR, str(MULTIFINGER_TYPE), '0.5', '0.5_'+str(round(best_indicators[0], 4))+'_'+str(round(best_indicators[1], 4))+'_'+str(round(best_indicators[2], 4))+'_'+str(epoch)+'.pth')
        torch.save(multifinger_net, model_path)
    if mean_indicators[3] - best_indicators[3] > 0.01 and mean_indicators[4] > 0.05 or (abs(mean_indicators[3] - best_indicators[3]) < 0.01 and mean_indicators[4] > best_indicators[4]):
        best_indicators[3:6] = mean_indicators[3:6].tolist()
        if not os.path.exists(os.path.join(LOG_DIR, str(MULTIFINGER_TYPE), '0.7')):
            os.makedirs(os.path.join(LOG_DIR, str(MULTIFINGER_TYPE), '0.7'))
        model_path = os.path.join(LOG_DIR, str(MULTIFINGER_TYPE), '0.7', '0.7_'+str(round(best_indicators[3], 4))+'_'+str(round(best_indicators[4], 4))+'_'+str(round(best_indicators[5], 4))+'_'+str(epoch)+'.pth')
        torch.save(multifinger_net, model_path)
    if mean_indicators[6] - best_indicators[6] > 0.01 and mean_indicators[7] > 0.05 or (abs(mean_indicators[6] - best_indicators[6]) < 0.01 and mean_indicators[7] > best_indicators[7]):
        best_indicators[6:9] = mean_indicators[6:9].tolist()
        if not os.path.exists(os.path.join(LOG_DIR, str(MULTIFINGER_TYPE), '0.9')):
            os.makedirs(os.path.join(LOG_DIR, str(MULTIFINGER_TYPE), '0.9'))
        model_path = os.path.join(LOG_DIR, str(MULTIFINGER_TYPE), '0.9', '0.9_'+str(round(best_indicators[6], 4))+'_'+str(round(best_indicators[7], 4))+'_'+str(round(best_indicators[8], 4))+'_'+str(epoch)+'.pth')       
        torch.save(multifinger_net, model_path)
    return best_indicators

def train_one_epoch():
    # adjust_learning_rate(optimizer, EPOCH_CNT)
    # bnm_scheduler.step() # decay BN momentum
    # set model to training mode
    multifinger_net.train()
    data_time = 0.
    net_time = 0.
    tic = time.time()
    all_losses = []
    weights = []
    pres = []
    recalls = []
    f1s = []
    indicator_detachs = []
    every_indicator_detachs = []
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        batch_data_label = convert_data_to_gpu(batch_data_label)
        if batch_data_label["result"].shape[0] == 1:
            continue
        weights.append(batch_data_label["result"].shape[0])
        toc = time.time()
        data_time += toc - tic
        tic = time.time()
        # Forward pass
        end_points = {}
        end_points["two_fingers_pose_angle_type"] = batch_data_label["two_fingers_pose_angle_type"]
        end_points["two_fingers_pose_depth_type"] = batch_data_label["two_fingers_pose_depth_type"]
        end_points["multifinger_pose_finger_type"] = batch_data_label["multifinger_pose_finger_type"]
        end_points["multifinger_pose_depth_type"] = batch_data_label["multifinger_pose_depth_type"] 

        end_points["grasp_preds_features"] = batch_data_label['grasp_preds_features']
        end_points["if_flip"] = batch_data_label['if_flip']

        grasp_preds, end_points = multifinger_net(end_points)
    
        end_points["result"] = batch_data_label["result"]
        loss, indicator, every_indicator = criterion(end_points)
        indicator_detach = []
        for item in indicator:
            indicator_detach.append(item.detach().item())
        indicator_detachs.append(indicator_detach)

        every_indicator_detach = []
        for acc_type in every_indicator:
            gripper_types = []
            for gripper_type in acc_type:
                accs = []
                for item in gripper_type:
                    accs.append(item.detach().item()) 
                gripper_types.append(accs)
            every_indicator_detach.append(gripper_types)
        every_indicator_detachs.append(every_indicator_detach)

        loss.backward()
        all_losses.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        toc = time.time()
        net_time += toc - tic

        batch_interval = 1000
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            log_string("Train loss:{}".format(loss))
            data_time = 0.
            net_time = 0.
        tic = time.time()

    weights = np.array(weights)
    weights = weights / np.sum(weights)
    mean_loss = np.sum(weights * np.array(all_losses))
    mean_indicators = np.sum(weights.reshape((-1,1)) * np.array(indicator_detachs), axis=0)
    mean_every_indicators = np.sum(weights.reshape((-1,1,1,1)) * np.array(every_indicator_detachs), axis=0)

    log_string("===== {} =====".format(MULTIFINGER_TYPE))
    log_string("Mean Train loss:{}".format(mean_loss))
    log_string("Mean Train @0.5 precision:{}, recall:{}, f1:{}".format(mean_indicators[0], mean_indicators[1], mean_indicators[2]))
    log_string("Mean Train @0.7 precision:{}, recall:{}, f1:{}".format(mean_indicators[3], mean_indicators[4], mean_indicators[5]))
    log_string("Mean Train @0.9 precision:{}, recall:{}, f1:{}".format(mean_indicators[6], mean_indicators[7], mean_indicators[8]))
    for mt in range(NUM_MULTIFINGER_TYPE):
        log_string("Mean Gripper Type:{}{}, Train @0.9 precision:{}, recall:{}, f1:{}, tp:{}".format(
            GRIPPER_TYPE, MULTIFINGER_TYPE, mean_every_indicators[2][mt][0], 
            mean_every_indicators[2][mt][1], mean_every_indicators[2][mt][2], mean_every_indicators[2][mt][3]))
    log_string("===== =====")
    TRAIN_WRITER.add_scalar("train/loss", mean_loss, EPOCH_CNT)
    TRAIN_WRITER.add_scalar("leraning rate", lr_scheduler.get_last_lr()[0], EPOCH_CNT)

    TRAIN_WRITER.add_scalar("train/precision_0.5", mean_indicators[0], EPOCH_CNT)
    TRAIN_WRITER.add_scalar("train/recall_0.5", mean_indicators[1], EPOCH_CNT)
    TRAIN_WRITER.add_scalar("train/f1_0.5", mean_indicators[2], EPOCH_CNT)
    TRAIN_WRITER.add_scalar("train/precision_0.7", mean_indicators[3], EPOCH_CNT)
    TRAIN_WRITER.add_scalar("train/recall_0.7", mean_indicators[4], EPOCH_CNT)
    TRAIN_WRITER.add_scalar("train/f1_0.7", mean_indicators[5], EPOCH_CNT)
    TRAIN_WRITER.add_scalar("train/precision_0.9", mean_indicators[6], EPOCH_CNT)
    TRAIN_WRITER.add_scalar("train/recall_0.9", mean_indicators[7], EPOCH_CNT)
    TRAIN_WRITER.add_scalar("train/f1_0.9", mean_indicators[8], EPOCH_CNT)
    for mt in range(NUM_MULTIFINGER_TYPE):
        TRAIN_WRITER.add_scalar("train_{}_precision/{}/precision_0.9".format(GRIPPER_TYPE, MULTIFINGER_TYPE), mean_every_indicators[2][mt][0], EPOCH_CNT)
        TRAIN_WRITER.add_scalar("train_{}_call/{}/recall_0.9".format(GRIPPER_TYPE, MULTIFINGER_TYPE), mean_every_indicators[2][mt][1], EPOCH_CNT)
        TRAIN_WRITER.add_scalar("train__tp{}/{}/tp_0.9".format(GRIPPER_TYPE, MULTIFINGER_TYPE), mean_every_indicators[2][mt][3], EPOCH_CNT)

def eval_one_epoch():
    # adjust_learning_rate(optimizer, EPOCH_CNT)
    # bnm_scheduler.step() # decay BN momentum
    # set model to training mode
    multifinger_net.eval()
    data_time = 0.
    net_time = 0.
    tic = time.time()
    all_losses = []
    weights = []
    pres = []
    recalls = []
    f1s = []
    indicator_detachs = []
    every_indicator_detachs = []
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        batch_data_label = convert_data_to_gpu(batch_data_label)
        weights.append(batch_data_label["result"].shape[0])
        toc = time.time()
        data_time += toc - tic
        tic = time.time()
        # Forward pass
        end_points = {}
        end_points["two_fingers_pose_angle_type"] = batch_data_label["two_fingers_pose_angle_type"]
        end_points["two_fingers_pose_depth_type"] = batch_data_label["two_fingers_pose_depth_type"]
        end_points["multifinger_pose_finger_type"] = batch_data_label["multifinger_pose_finger_type"]
        end_points["multifinger_pose_depth_type"] = batch_data_label["multifinger_pose_depth_type"] 

        end_points["grasp_preds_features"] = batch_data_label['grasp_preds_features']
        end_points["if_flip"] = batch_data_label['if_flip']

        grasp_preds, end_points = multifinger_net(end_points)
     
        end_points["result"] = batch_data_label["result"]
        loss, indicator, every_indicator = criterion(end_points)
        indicator_detach = []
        for item in indicator:
            indicator_detach.append(item.detach().item())
        indicator_detachs.append(indicator_detach)

        every_indicator_detach = []
        for acc_type in every_indicator:
            gripper_types = []
            for gripper_type in acc_type:
                accs = []
                for item in gripper_type:
                    accs.append(item.detach().item()) 
                gripper_types.append(accs)
            every_indicator_detach.append(gripper_types)
        every_indicator_detachs.append(every_indicator_detach)

        all_losses.append(loss.item())

    weights = np.array(weights)
    weights = weights / np.sum(weights)
    mean_loss = np.sum(weights * np.array(all_losses))
    mean_indicators = np.sum(weights.reshape((-1,1)) * np.array(indicator_detachs), axis=0)
    mean_every_indicators = np.sum(weights.reshape((-1,1,1,1)) * np.array(every_indicator_detachs), axis=0)

    log_string("===== {} =====".format(MULTIFINGER_TYPE))
    log_string("Mean Test loss:{}".format(mean_loss))
    log_string("Mean Test @0.5 precision:{}, recall:{}, f1:{}".format(mean_indicators[0], mean_indicators[1], mean_indicators[2]))
    log_string("Mean Test @0.7 precision:{}, recall:{}, f1:{}".format(mean_indicators[3], mean_indicators[4], mean_indicators[5]))
    log_string("Mean Test @0.9 precision:{}, recall:{}, f1:{}".format(mean_indicators[6], mean_indicators[7], mean_indicators[8]))
    for mt in range(NUM_MULTIFINGER_TYPE):
        log_string("Mean Gripper Type:{}{}, Test @0.9 precision:{}, recall:{}, f1:{}, tp:{}".format(
            GRIPPER_TYPE, MULTIFINGER_TYPE, mean_every_indicators[2][mt][0], 
            mean_every_indicators[2][mt][1], mean_every_indicators[2][mt][2], mean_every_indicators[2][mt][3]))
    log_string("===== =====")
    TEST_WRITER.add_scalar("test/loss", mean_loss, EPOCH_CNT)
 
    TEST_WRITER.add_scalar("test/precision_0.5", mean_indicators[0], EPOCH_CNT)
    TEST_WRITER.add_scalar("test/recall_0.5", mean_indicators[1], EPOCH_CNT)
    TEST_WRITER.add_scalar("test/f1_0.5", mean_indicators[2], EPOCH_CNT)
    TEST_WRITER.add_scalar("test/precision_0.7", mean_indicators[3], EPOCH_CNT)
    TEST_WRITER.add_scalar("test/recall_0.7", mean_indicators[4], EPOCH_CNT)
    TEST_WRITER.add_scalar("test/f1_0.7", mean_indicators[5], EPOCH_CNT)
    TEST_WRITER.add_scalar("test/precision_0.9", mean_indicators[6], EPOCH_CNT)
    TEST_WRITER.add_scalar("test/recall_0.9", mean_indicators[7], EPOCH_CNT)
    TEST_WRITER.add_scalar("test/f1_0.9", mean_indicators[8], EPOCH_CNT)
    for mt in range(NUM_MULTIFINGER_TYPE):
        TEST_WRITER.add_scalar("test_{}_precision/{}/precision_0.9".format(GRIPPER_TYPE, MULTIFINGER_TYPE), mean_every_indicators[2][mt][0], EPOCH_CNT)
        TEST_WRITER.add_scalar("test_{}_recall/{}/recall_0.9".format(GRIPPER_TYPE, MULTIFINGER_TYPE), mean_every_indicators[2][mt][1], EPOCH_CNT)
        TEST_WRITER.add_scalar("test_{}_tp/{}/tp_0.9".format(GRIPPER_TYPE, MULTIFINGER_TYPE), mean_every_indicators[2][mt][3], EPOCH_CNT)
    return mean_indicators

def train(start_epoch):
    global EPOCH_CNT 
    min_loss = 1e10
    loss = 0
    best_indicators = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
    torch.set_printoptions(5)
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** TRAIN EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f'%(lr_scheduler.get_last_lr()[0]))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        # Train
        np.random.seed()
        train_one_epoch()
        log_string(" ---- Evaluating one epoch ---- ")
        mean_indicators = eval_one_epoch()
        lr_scheduler.step()
        best_indicators = save_model(epoch, mean_indicators, best_indicators)
        
        print('best indicators: ', best_indicators)



if __name__=='__main__':
    train(start_epoch)
