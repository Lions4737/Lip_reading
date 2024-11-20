#既存のモデルに対して、predictとtruthの'sil'と'pau'を除く文字一致に対して報酬を与える
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
from torch.utils.data import DataLoader
from model import LipNet
from dataset4 import MyDataset
from tensorboardX import SummaryWriter

# 最小のresultxxx.txtを見つける関数
def find_available_result_file():
    for i in range(1000):
        filename = f"result{str(i).zfill(3)}.txt"
        if not os.path.exists(filename):
            return filename
    raise RuntimeError("No available resultxxx.txt slot found")

# ログファイルを決定
log_file = find_available_result_file()

# ログに書き込む関数
def log(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")

if __name__ == '__main__':
    log("hello1")
    opt = __import__('options')
    log("hello2")
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    log("hello3")    
    writer = SummaryWriter()
    log("hello4")

# その他関数の内部printをlogに置き換え
# 以下一部を例として変更
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

def calculate_ce_loss_with_reward(y, txt, vid_len, txt_len, sil_index, pau_index, reward=1.0):
    """
    クロスエントロピー損失に特定の条件で報酬を加味する。
    y: モデルの出力 (batch_size, seq_len, num_classes)
    txt: ターゲット音素列 (batch_size, seq_len)
    vid_len: 予測フレームの長さ (batch_size)
    txt_len: 音素列の長さ (batch_size)
    sil_index: 'sil' のインデックス
    pau_index: 'pau' のインデックス
    reward: 報酬の強さ（正しい予測時に損失を減らす量）
    """
    log_probs = F.log_softmax(y, dim=-1)  # 対数確率を計算
    batch_size, seq_len, num_classes = log_probs.shape

    # 損失の初期計算
    loss = F.nll_loss(log_probs.view(-1, num_classes), txt.view(-1), ignore_index=-1, reduction='none')
    loss = loss.view(batch_size, seq_len)  # (batch_size, seq_len)

    # 特定の条件で報酬を加味
    predictions = y.argmax(dim=-1)  # モデルの予測 (batch_size, seq_len)
    reward_mask = (txt != sil_index) & (txt != pau_index) & (predictions == txt)  # 条件に合致するマスク
    loss = loss - reward * reward_mask.float()  # 報酬を加算 (損失を減らす)

    # 有効な長さ（vid_len, txt_len）のみ考慮
    mask = torch.arange(seq_len).unsqueeze(0).cuda() < vid_len.unsqueeze(1)
    loss = loss * mask.float()

    return loss.sum() / mask.float().sum()  # 有効な部分のみ平均化


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
    vid = input.get('vid')
    txt = input.get('txt')
    vid_len = input.get('vid_len')
    txt_len = input.get('txt_len')

    if not torch.all(vid_len >= txt_len):
        log("Error: vid_len < txt_len detected!")
        for i in range(len(vid_len)):
            if vid_len[i] < txt_len[i]:
                log(f"Batch {i}: vid_len = {vid_len[i].item()}, txt_len = {txt_len[i].item()}")

    if torch.any(vid_len <= 0) or torch.any(txt_len <= 0):
        log("Error: vid_len or txt_len <= 0 detected!")
        log("vid_len <= 0: " + str(vid_len[vid_len <= 0]))
        log("txt_len <= 0: " + str(txt_len[txt_len <= 0]))

def test(model, net):

    with torch.no_grad():
        dataset = MyDataset(opt.video_path,
            opt.anno_path,
            opt.val_list,
            opt.vid_padding,
            opt.txt_padding,
            'test')
            
        log('num_test_data:{}'.format(len(dataset.data)))  
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
            #loss = calculate_ce_loss(y, txt, vid_len, txt_len).detach().cpu().numpy()
            sil_index = dataset.sil_index
            pau_index = dataset.pau_index
            loss = calculate_ce_loss_with_reward(y, txt, vid_len, txt_len, sil_index, pau_index, reward=1.0)
            loss_list.append(loss)
            pred_txt = ctc_decode(y)
            
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            wer.extend(MyDataset.wer(pred_txt, truth_txt)) 
            cer.extend(MyDataset.cer(pred_txt, truth_txt))              
            if(i_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(i_iter+1)
                eta = v * (len(loader)-i_iter) / 3600.0
                
                log(''.join(101*'-'))                
                log('{:<50}|{:>50}'.format('predict', 'truth'))
                log(''.join(101*'-'))                
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
                    log('{:<50}|{:>50}'.format(predict, truth))                
                log(''.join(101 *'-'))
                log('test_iter={},eta={},wer={},cer={}'.format(i_iter,eta,np.array(wer).mean(),np.array(cer).mean()))                
                log(''.join(101 *'-'))
                
        #return (np.array(loss_list).mean(), np.array(wer).mean(), np.array(cer).mean())
        return (
    np.array([loss.cpu().item() for loss in loss_list]).mean(),
    np.array(wer).mean(),
    np.array(cer).mean()
)


def train(model, net):
    log("Starting training...")
    dataset = MyDataset(opt.video_path, opt.anno_path, opt.train_list, opt.vid_padding, opt.txt_padding, 'train')
    loader = dataset2dataloader(dataset) 
    optimizer = optim.Adam(model.parameters(), lr=opt.base_lr, weight_decay=0., amsgrad=True)
    log(f'num_train_data: {len(dataset.data)}')    
    tic = time.time()

    torch.cuda.empty_cache()
    train_cer = []
    for epoch in range(opt.max_epoch):
        for (i_iter, input) in enumerate(loader):
            debug_batch(input)
            model.train()
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()

            optimizer.zero_grad()
            y = net(vid)
            sil_index = dataset.sil_index
            pau_index = dataset.pau_index
            loss = calculate_ce_loss_with_reward(y, txt, vid_len, txt_len, sil_index, pau_index, reward=1.0)
            #loss = calculate_ce_loss(y, txt, vid_len, txt_len)
            # 損失チェック
            log(f"nan/inf check for loss:")
            log(f"Loss contains NaN: {torch.isnan(loss).any().item()}")
            log(f"Loss contains Inf: {torch.isinf(loss).any().item()}")
            for name, param in model.named_parameters():
                if torch.any(torch.isnan(param)) or torch.any(torch.isinf(param)):
                    log(f"NaN or Inf detected in parameter: {name}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 勾配チェック
            for name, param in model.named_parameters():
                if param.grad is not None:
                    log(f"Gradient {name}: Mean = {param.grad.mean()}, Std = {param.grad.std()}")
                    if torch.any(torch.isnan(param.grad)):
                        log(f"NaN detected in gradient of {name}")
                    if torch.any(torch.isinf(param.grad)):
                        log(f"Inf detected in gradient of {name}")
            if opt.is_optimize:
                optimizer.step()
            tot_iter = i_iter + epoch * len(loader)
            pred_txt = ctc_decode(y)
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_cer.extend(MyDataset.cer(pred_txt, truth_txt))

            if tot_iter % opt.display == 0:
                v = (time.time() - tic) / (tot_iter + 1)
                eta = (len(loader) - i_iter) * v / 3600.0
                writer.add_scalar('train loss', loss, tot_iter)
                writer.add_scalar('train cer', np.array(train_cer).mean(), tot_iter)  
                log(''.join(101 * '-'))                
                log('{:<50}|{:>50}'.format('predict', 'truth'))                
                log(''.join(101 * '-'))
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
                    log('{:<50}|{:>50}'.format(predict, truth))
                log(''.join(101 * '-'))                
                log(f'epoch={epoch}, tot_iter={tot_iter}, eta={eta}, loss={loss}, train_cer={np.array(train_cer).mean()}')
                log(''.join(101 * '-'))
            if(tot_iter % opt.test_step == 0):                
                (loss, wer, cer) = test(model, net)
                log('i_iter={},lr={},loss={},wer={},cer={}'
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

# main部分も同様に修正
if __name__ == '__main__':
    log("Loading options...")
    model = LipNet().cuda()
    net = nn.DataParallel(model).cuda()

    if hasattr(opt, 'weights'):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if k not in pretrained_dict.keys()]
        log(f'loaded params/tot params: {len(pretrained_dict)}/{len(model_dict)}')
        log(f'miss matched params: {missed_params}')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)
    train(model, net)
