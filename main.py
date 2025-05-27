import argparse
import time
import numpy as np
import torch
import os
from utilis_func import *
from trainer import Trainer
from TFFN_RGAT import STRGAT
import sys
sys.argv = [' ']

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def setup_device(device_str):
    if device_str == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def build_model(args, device):
    return STRGAT(
        args.gcn_true, 
        args.buildA_true, 
        args.gcn_depth, 
        args.num_nodes, 
        device, 
        dropout=args.dropout, 
        conv_channels=args.conv_channels, 
        residual_channels=args.residual_channels,
        skip_channels=args.skip_channels, 
        end_channels=args.end_channels,
        seq_length=args.seq_in_len, 
        in_dim=args.in_dim, 
        out_dim=args.seq_out_len,
        layers=args.layers, 
        propalpha=args.propalpha, 
        layer_norm_affline=True
    )

def train_epoch(engine, train_loader, device, print_every):
    train_loss = []
    train_mape = []
    train_rmse = []
    for iter, (x, y) in enumerate(train_loader.get_iterator()):
        trainx = torch.Tensor(x).to(device).transpose(1, 3)
        trainy = torch.Tensor(y).to(device).transpose(1, 3)[:, 0, :, :]
        metrics = engine.train(trainx, trainy)
        train_loss.append(metrics[0])
        train_mape.append(metrics[1])
        train_rmse.append(metrics[2])
        if iter % print_every == 0:
            log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
            print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
    return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)

def validate_epoch(engine, val_loader, device):
    valid_loss = []
    valid_mape = []
    valid_rmse = []
    for iter, (x, y) in enumerate(val_loader.get_iterator()):
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        testy = torch.Tensor(y).to(device).transpose(1, 3)[:, 0, :, :]
        metrics = engine.eval(testx, testy)
        valid_loss.append(metrics[0])
        valid_mape.append(metrics[1])
        valid_rmse.append(metrics[2])
    return np.mean(valid_loss), np.mean(valid_mape), np.mean(valid_rmse)

def predict_and_save(engine, data_loader, scaler, device, scaling_required, save_prefix, runid):
    outputs = []
    realy = torch.Tensor(data_loader['y']).to(device).transpose(1, 3)[:, 0, :, :]
    for iter, (x, _) in enumerate(data_loader['loader'].get_iterator()):
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            preds, _ = engine.pred(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds)
    yhat = torch.cat(outputs, dim=0)[:realy.size(0)]
    if scaling_required:
        pred = scaler.inverse_transform(yhat)
    else:
        pred = yhat
    np.save(f"{save_prefix}_{runid}_pred.npy", pred.squeeze().cpu().numpy())
    np.save(f"{save_prefix}_{runid}_label.npy", realy.squeeze().cpu().numpy())

def main(runid, args):
    device = setup_device(args.device)
    torch.manual_seed(runid)
    torch.cuda.manual_seed_all(runid)
    np.random.seed(runid)
    os.environ['PYTHONHASHSEED'] = str(runid)
    
    dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, args.scaling_required)
    scaler = dataloader['scaler']
    model = build_model(args, device)
    print(args)
    print('The receptive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)
    
    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, scaler, device, args.scaling_required)
    
    his_loss = []
    val_time = []
    train_time = []
    minl = 1e5
    
    for i in range(1, args.epochs+1):
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        train_loss, train_mape, train_rmse = train_epoch(engine, dataloader['train_loader'], device, args.print_every)
        train_time.append(time.time()-t1)
        
        s1 = time.time()
        val_loss, val_mape, val_rmse = validate_epoch(engine, dataloader['val_loader'], device)
        val_time.append(time.time()-s1)
        
        his_loss.append(val_loss)
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, train_loss, train_mape, train_rmse, val_loss, val_mape, val_rmse, (time.time()-t1)),flush=True)
        
        if val_loss < minl:
            torch.save(engine.model.state_dict(), args.save + "exp" + str(args.expid) + "_" + str(runid) + ".pth")
            minl = val_loss
    
    engine.model.load_state_dict(torch.load(args.save + "exp" + str(args.expid) + "_" + str(runid) + ".pth"))
    
    save_prefix = os.path.join(args.save, f'exp{args.expid}')
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    
    for data_type in ['train', 'val', 'test']:
        loader = dataloader[f'{data_type}_loader']
        y_data = dataloader[f'y_{data_type}']
        data_dict = {'loader': loader, 'y': y_data}
        predict_and_save(engine, data_dict, scaler, device, args.scaling_required, f"{save_prefix}_{data_type}", runid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data and Pre-processing
    parser.add_argument('--device', type=str, default='cpu', help='')
    parser.add_argument('--data', type=str, default='./data/my-data', help='data path')
    parser.add_argument('--scaling_required', type=bool, default=False, help='Whether to scale input for model and inverse scale output from model.')
    parser.add_argument('--save', type=str, default='./save/', help='save path')
    parser.add_argument('--expid', type=str, default='', help='experiment id')
    parser.add_argument('--runs', type=int, default=1, help='number of runs')
    parser.add_argument('--save_result',type=str,default='true',help='path to save forecasting results')
    # Training and optimization
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--clip', type=int, default=10, help='clip')
    parser.add_argument('--step_size1', type=int, default=2500, help='step_size')
    parser.add_argument('--epochs', type=int, default=1, help='')
    parser.add_argument('--print_every', type=int, default=5000, help='')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    ## CST-GNN Framework hyper-parameters
    parser.add_argument('--buildA_true', type=str_to_bool, default=True, help='whether to construct adaptive adjacency matrix')
    parser.add_argument('--propalpha', type=float, default=0.1, help='prop alpha in graph module')
    parser.add_argument('--num_nodes', type=int, default=16, help='number of nodes/variables')
    parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
    parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
    parser.add_argument('--dilation_exponential', type=int, default=1, help='dilation exponential')
    parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
    parser.add_argument('--residual_channels', type=int, default=16, help='residual channels')
    parser.add_argument('--skip_channels', type=int, default=32, help='skip channels')
    parser.add_argument('--end_channels', type=int, default=64, help='end channels')
    parser.add_argument('--layers', type=int, default=2, help='number of layers')
    parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
    parser.add_argument('--seq_in_len', type=int, default=100, help='input sequence length')
    parser.add_argument('--seq_out_len', type=int, default=1, help='output sequence length')  
    
    args = parser.parse_args()
    torch.set_num_threads(4)
    
    for i in range(args.runs):
        main(i, args)