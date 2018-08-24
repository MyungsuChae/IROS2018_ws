import torch, torchvision
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import numpy as np

from dataset import IEMOCAP, collate_fn
from model import MultiNet, AudioNet, VideoNet
from process import sec2hms
from tqdm import tqdm
import argparse
import time, os
st = time.time()

# Training settings
parser = argparse.ArgumentParser(description='Multimodal Multitask Emotion Recognition')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--valid-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 2)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--subset', type=str, nargs='+', default='',
                    help='Specify subset condition `toy`, `impro`, `session1`')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-save-interval', type=int, default=24, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--valid-interval', type=int, default=12, metavar='N',
                    help='how many batches to wait before logging valid status')
parser.add_argument('--modal', type=str, 
                    help='{audio | video | multi (default)}')
parser.add_argument('--loss', type=str, 
                    help='{Joint | Static}')
parser.add_argument('--stl_dim', type=int, default=-1, 
                    help='{-1: Equivalent on all task, 0: Emotion, 1: Gender')
args = parser.parse_args()

args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def getdate():
    import time
    return time.strftime('%y%m%d')

savepath = os.path.join('result', '{}_{}_{}'.format(getdate(), args.modal, args.loss))
if not os.path.exists(savepath):
    os.makedirs(savepath)
else:
    input("Path already exists, wish to continue?")
    os.system("rm -rf {}/*".format(savepath))

trainlosspath = os.path.join(savepath, 'train_loss.log')
validlosspath = os.path.join(savepath, 'valid_loss.log')
testlosspath  = os.path.join(savepath, 'test_loss.log')
trainaccpath = os.path.join(savepath, 'train_acc.log')
modelpath = os.path.join(savepath, 'model.pt')
bestmodelpath = os.path.join(savepath, 'model_best.pt')

min_loss = 1e8

def train(epoch):
    model.train()
    for batch_idx, (video, audio, video_lengths, emo, gen) in enumerate(train_loader):
        current_it = epoch * len(train_loader.dataset) + batch_idx * len(video)

        if args.cuda:
            video = video.cuda()
            audio = audio.cuda()
            emo = emo.cuda()
            gen = gen.cuda()
            video_lengths = video_lengths.cuda()

        video = Variable(video, requires_grad=False)
        audio = Variable(audio, requires_grad=False)
        emo = Variable(emo, requires_grad=False)
        gen = Variable(gen, requires_grad=False)

        optimizer.zero_grad()

        pred_emo, pred_gen = model(video, audio, video_lengths)
        loss = model.calc_loss((pred_emo, pred_gen), (emo, gen)).mean()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.) 
        optimizer.step()

        with open(trainlosspath, 'a') as f:
            f.write('{}\t{}\n'.format(current_it, loss.data))
        with open(trainaccpath, 'a') as f:
            pred_emo_arg = torch.argmax(pred_emo, 1)
            pred_gen_arg = torch.argmax(pred_gen, 1)
            acc_emo = float(sum(pred_emo_arg==emo))/len(emo) * 100
            acc_gen = float(sum(pred_gen_arg==gen))/len(emo) * 100
            f.write('{}\t{}\t{}\n'.format(current_it, acc_emo, acc_gen))

        if batch_idx % args.log_interval == 0:
            stdout = '[{}] Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(            
                sec2hms(time.time()-st), epoch, batch_idx * len(video), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data)

        if batch_idx % args.model_save_interval == 0:
            print(stdout)
            torch.save(model.state_dict(), modelpath)

def valid(epoch):
    global min_loss
    model.eval()
    valid_loss = 0
    for valid_it, (video, audio, video_lengths, emo, gen) in enumerate(tqdm(valid_loader)):
        if args.cuda:
            video, audio, emo, gen = video.cuda(), audio.cuda(), emo.cuda(), gen.cuda()

        video = Variable(video, requires_grad=False)
        audio = Variable(audio, requires_grad=False)
        emo = Variable(emo, requires_grad=False)
        gen = Variable(gen, requires_grad=False)

        pred_emo, pred_gen = model(video, audio, video_lengths)
        loss= model.calc_loss((pred_emo, pred_gen), (emo, gen))

        valid_loss += float(loss.mean().data.cpu())
         
    valid_loss /= len(valid_loader)
    with open(validlosspath, 'a') as f:
        f.write('{}\t{}\n'.format(epoch, valid_loss))
    if min_loss > loss.data:
        torch.save(model.state_dict(), bestmodelpath)
        print('best saved')
        min_loss = loss.data
    print('[{}] Valid Loss : {:.6f}'.format(sec2hms(time.time()-st), valid_loss))
            
def test(epoch):
    model.eval()
    test_loss = 0
    for test_it, (video, audio, video_lengths, emo, gen) in enumerate(tqdm(test_loader)):
        if args.cuda:
            video, audio, emo, gen = video.cuda(), audio.cuda(), emo.cuda(), gen.cuda()

        video = Variable(video, requires_grad=False)
        audio = Variable(audio, requires_grad=False)
        emo = Variable(emo, requires_grad=False)
        gen = Variable(gen, requires_grad=False)

        pred_emo, pred_gen = model(video, audio, video_lengths)
        loss= model.calc_loss((pred_emo, pred_gen), (emo, gen))

        test_loss += float(loss.mean().data.cpu())
    test_loss /= len(test_loader)
    with open(testlosspath, 'a') as f:
        f.write('{}\t{}\n'.format(epoch, test_loss))
    print('[{}] test Loss : {:.6f}'.format(sec2hms(time.time()-st), test_loss))

if __name__=='__main__':
    model = MultiNet(modal=args.modal, loss=args.loss, stl_dim = args.stl_dim)

    if args.cuda:
        model.cuda()

    train_loader = torch.utils.data.DataLoader(
        IEMOCAP(which_set='train', datatype=args.modal, subset=args.subset), 
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, **loader_kwargs)
    valid_loader = torch.utils.data.DataLoader(
        IEMOCAP(which_set='valid', datatype=args.modal, subset=args.subset), 
        batch_size=args.valid_batch_size, shuffle=True, collate_fn=collate_fn, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(
        IEMOCAP(which_set='test', datatype=args.modal, subset=args.subset), 
        batch_size=args.valid_batch_size, shuffle=True, collate_fn=collate_fn, **loader_kwargs)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10)

    for epoch in range(args.epochs): 
        scheduler.step()
        valid(epoch)
        test(epoch)
        train(epoch)
