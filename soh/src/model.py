import torch, torch.nn as nn, torch.nn.functional as F

class TBlock(nn.Module):
    def __init__(s, c, k=3, d=1, p=0.2):
        super().__init__()
        pad = (k-1)*d
        s.c1 = nn.Conv1d(c, c, k, padding=pad, dilation=d); s.b1 = nn.BatchNorm1d(c)
        s.c2 = nn.Conv1d(c, c, k, padding=pad, dilation=d); s.b2 = nn.BatchNorm1d(c)
        s.dp = nn.Dropout(p)
    def forward(s, x):
        L = x.size(-1)                                   # causal: left pad, cut tail
        y = s.dp(F.relu(s.b1(s.c1(x)[..., :L])))
        y = s.dp(F.relu(s.b2(s.c2(y)[..., :L])))
        return F.relu(x + y)

class SOHNet(nn.Module):
    def __init__(s, in_ch=3, c=32, h=64, p=0.2, dil=(1,2,4,8)):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv1d(in_ch, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU())
        s.tcn = nn.Sequential(*[TBlock(c, 3, d, p) for d in dil])
        s.lstm = nn.LSTM(c, h, batch_first=True)
        s.aw = nn.Linear(h, h); s.av = nn.Linear(h, 1, bias=False)   # additive attn
        s.dp = nn.Dropout(p); s.fc = nn.Linear(h, 1)
        for m in s.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(s, x, ret_attn=False):
        z = s.tcn(s.stem(x)).transpose(1, 2)            # B,L,c
        o, _ = s.lstm(z)                                 # B,L,h
        a = torch.softmax(s.av(torch.tanh(s.aw(o))).squeeze(-1), dim=1)   # B,L
        ctx = (a.unsqueeze(-1) * o).sum(1)
        y = s.fc(s.dp(ctx)).squeeze(-1)
        return (y, a) if ret_attn else y
