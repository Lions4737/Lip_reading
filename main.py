import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
from torch.utils.data import DataLoader
from model import LipNet
from dataset import MyDataset
from tensorboardX import SummaryWriter

if(__name__ == '__main__'):
    opt = __import__('options')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu    
    writer = SummaryWriter()

# クロスエントロピー損失を計算する関数
def calculate_ce_loss(y, txt, vid_len, txt_len):
    """
    予測フレーム数と音素列の長さが一致している場合に1対1で比較するためのクロスエントロピー損失計算。
    y: モデルの出力 (batch_size, seq_len, num_classes)
    txt: ターゲット音素列 (batch_size, seq_len)
    vid_len: 予測フレームの長さ (batch_size)
    txt_len: 音素列の長さ (batch_size)
    """
    # log_softmaxを使って対数確率を計算
    log_probs = F.log_softmax(y, dim=-1)  # (batch_size, seq_len, num_classes)

    # クロスエントロピー損失を計算
    # ここでは、各フレームの予測とターゲットが1対1で対応しているので、各フレームのクロスエントロピーを計算
    loss = F.nll_loss(log_probs.view(-1, log_probs.size(-1)), txt.view(-1), ignore_index=-1)
    
    return loss

def dataset2dataloader(dataset, num_workers=0, shuffle=True): #元々 num_workers=opt.num_workers,
    return DataLoader(dataset,
        batch_size = opt.batch_size, 
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = False)

def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return np.array(lr).mean()  

def ctc_decode(y):
    result = []
    y = y.argmax(-1)
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]

def debug_batch(input):
    """
    データの不整合と異常を確認するデバッグ関数。
    """
    vid = input.get('vid')  # ビデオデータ
    txt = input.get('txt')  # テキストデータ
    vid_len = input.get('vid_len')  # ビデオシーケンスの長さ
    txt_len = input.get('txt_len')  # テキストシーケンスの長さ

    # 1. (a) vid_len と txt_len の不整合チェック
    '''print("vid shape:", vid.shape)  # (batch_size, channels, frames, height, width)
    print("txt shape:", txt.shape)  # (batch_size, seq_length)
    print("vid_len:", vid_len)  # 各シーケンスの長さ
    print("txt_len:", txt_len)  # 各テキストの長さ
    print("About vid")
    print("video size:", vid.size())
    print(vid)
    print("About txt")
    print("text size:", txt.size())
    print(txt)'''

    # vid_len >= txt_len を確認
    if not torch.all(vid_len >= txt_len):
        print("Error: vid_len < txt_len detected!")
        for i in range(len(vid_len)):
            if vid_len[i] < txt_len[i]:
                print(f"Batch {i}: vid_len = {vid_len[i].item()}, txt_len = {txt_len[i].item()}")

    # vid_len または txt_len が 0 以下になっていないか確認
    if torch.any(vid_len <= 0) or torch.any(txt_len <= 0):
        print("Error: vid_len or txt_len <= 0 detected!")
        print("vid_len <= 0:", vid_len[vid_len <= 0])
        print("txt_len <= 0:", txt_len[txt_len <= 0])

    '''# 1. (b) ビデオとテキストデータそのものの異常チェック
    print("Checking video tensor for NaN/Inf...")
    print("Video tensor NaN:", torch.any(torch.isnan(vid)))
    print("Video tensor Inf:", torch.any(torch.isinf(vid)))

    print("Checking text tensor for NaN/Inf...")
    print("Text tensor NaN:", torch.any(torch.isnan(txt)))
    print("Text tensor Inf:", torch.any(torch.isinf(txt)))'''
    
def test(model, net):

    with torch.no_grad():
        dataset = MyDataset(opt.video_path,
            opt.anno_path,
            opt.val_list,
            opt.vid_padding,
            opt.txt_padding,
            'test')
            
        print('num_test_data:{}'.format(len(dataset.data)))  
        model.eval()
        loader = dataset2dataloader(dataset, shuffle=False)
        loss_list = []
        wer = []
        cer = []
        crit = nn.CTCLoss()
        tic = time.time()
        for (i_iter, input) in enumerate(loader):            
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()
            
            y = net(vid)
            
            # クロスエントロピー損失を計算
            loss = calculate_ce_loss(y, txt, vid_len, txt_len).detach().cpu().numpy()
            loss_list.append(loss)
            pred_txt = ctc_decode(y)
            
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            wer.extend(MyDataset.wer(pred_txt, truth_txt)) 
            cer.extend(MyDataset.cer(pred_txt, truth_txt))              
            if(i_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(i_iter+1)
                eta = v * (len(loader)-i_iter) / 3600.0
                
                print(''.join(101*'-'))                
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-'))                
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
                    print('{:<50}|{:>50}'.format(predict, truth))                
                print(''.join(101 *'-'))
                print('test_iter={},eta={},wer={},cer={}'.format(i_iter,eta,np.array(wer).mean(),np.array(cer).mean()))                
                print(''.join(101 *'-'))
                
        return (np.array(loss_list).mean(), np.array(wer).mean(), np.array(cer).mean())
    
def train(model, net):
    
    dataset = MyDataset(opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'train')
        
    loader = dataset2dataloader(dataset) 
    optimizer = optim.Adam(model.parameters(),
                lr = opt.base_lr,
                weight_decay = 0.,
                amsgrad = True)
                
    print('num_train_data:{}'.format(len(dataset.data)))    
    tic = time.time()
    
    torch.cuda.empty_cache()
    train_cer = []
    for epoch in range(opt.max_epoch):
        for (i_iter, input) in enumerate(loader):
            # データローダー出力チェック
            '''print("Checking DataLoader output for NaN/Inf...")
            print("Video tensor NaN:", torch.any(torch.isnan(input['vid'])))
            print("Video tensor Inf:", torch.any(torch.isinf(input['vid'])))
            print("Text tensor NaN:", torch.any(torch.isnan(input['txt'])))
            print("Text tensor Inf:", torch.any(torch.isinf(input['txt'])))'''
            debug_batch(input)
            model.train()
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()
            
            optimizer.zero_grad()
            y = net(vid)
            
            # クロスエントロピー損失を計算
            loss = calculate_ce_loss(y, txt, vid_len, txt_len)
            # 損失チェック
            print("Loss NaN:", torch.isnan(loss))
            print("Loss Inf:", torch.isinf(loss))
            for name, param in model.named_parameters():
                if torch.any(torch.isnan(param)) or torch.any(torch.isinf(param)):
                    print(f"NaN or Inf detected in parameter: {name}")

            loss.backward()
            # 勾配のクリッピングを追加
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 勾配チェック
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Gradient {name}: Mean = {param.grad.mean()}, Std = {param.grad.std()}")
                    if torch.any(torch.isnan(param.grad)):
                        print(f"NaN detected in gradient of {name}")
                    if torch.any(torch.isinf(param.grad)):
                        print(f"Inf detected in gradient of {name}")

            if(opt.is_optimize):
                optimizer.step()
            
            tot_iter = i_iter + epoch*len(loader)
            
            pred_txt = ctc_decode(y)
            
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_cer.extend(MyDataset.cer(pred_txt, truth_txt))
            
            if(tot_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(tot_iter+1)
                eta = (len(loader)-i_iter)*v/3600.0
                
                writer.add_scalar('train loss', loss, tot_iter)
                writer.add_scalar('train cer', np.array(train_cer).mean(), tot_iter)              
                print(''.join(101*'-'))                
                print('{:<50}|{:>50}'.format('predict', 'truth'))                
                print(''.join(101*'-'))
                
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101*'-'))                
                print('epoch={},tot_iter={},eta={},loss={},train_cer={}'.format(epoch, tot_iter, eta, loss, np.array(train_cer).mean()))
                print(''.join(101*'-'))
                
            if(tot_iter % opt.test_step == 0):                
                (loss, wer, cer) = test(model, net)
                print('i_iter={},lr={},loss={},wer={},cer={}'
                    .format(tot_iter,show_lr(optimizer),loss,wer,cer))
                writer.add_scalar('val loss', loss, tot_iter)                    
                writer.add_scalar('wer', wer, tot_iter)
                writer.add_scalar('cer', cer, tot_iter)
                savename = '{}_loss_{}_wer_{}_cer_{}.pt'.format(opt.save_prefix, loss, wer, cer)
                (path, name) = os.path.split(savename)
                if(not os.path.exists(path)): os.makedirs(path)
                torch.save(model.state_dict(), savename)
                if(not opt.is_optimize):
                    exit()

# main部分を変更し、ネットワークをトレーニング
if(__name__ == '__main__'):
    print("Loading options...")
    model = LipNet()
    model = model.cuda()
    net = nn.DataParallel(model).cuda()

    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)
    train(model, net)
