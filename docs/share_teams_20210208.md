# 2021/02/08

## モデルと提出ファイルの共有



|model|lwlap|preci|recall|LB|memo|Drive|
|--|--|--|--|--|--|--|
|resnet18|0.9589|0.657||0.942|現状SingleでBESTだが、出力値が低い||
|resnet18|0.9693|0.9590|0.7336|0.936|現状ベストプラクティス★||
|resnet18|0.9457|0.5744|0.08893|0.921|windowサイズ256だが、勾配累積を使っていないので出力値が低い||
|resnet18|0.9657|0.9665|0.7125||CBAM||
|resnest50|0.9397|0.8985||0.935|現状ベストプラクティス★||
|densener121|0.9392|0.8706||0.926|現状ベストプラクティス★||
|efficientnet_b0|0.8897|0.8748||0.922|現状ベストプラクティス★||
|efficientnet_b0|0.9002|0.8758|||↑のパラメータ調整||

- submitしていなかったり、実装が間に合わずスコアを計算していないものもある
- precision/recallは閾値0.5で計算している
- 現状ベストプラクティス
  - 明確にラベルが付いていないlossを計算しないのとpseudo labelingのcycle
  - 勾配累積
  - mixup lastlayer
  - positive lossをバッチの中で重み減衰
- アンサンブルは★がついている4個の単純平均でLB=0.941


### モデル

- 基本これで読み込める
- efficientnet_b0のヘッダー名が`resnet_head`になるので注意(私が仕込んだバグ)
- CBAMを使う際は`# CBAM用`のコメントの部分を有効にする

```python
# ヘッダーのインデックスと次元数
MODEL_HEADER_INFO = {
    "resnet18": (-2, 512),
    "densenet121": (-2, 1024),
    "efficientnet_b0": (-5, 320),
    "resnest50d": (-2, 2048),
    "mobilenetv2_100": (-2, 1280),
}

def interpolate(x: torch.Tensor, ratio: int):
    x = x.transpose(1, 2)
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    upsampled = upsampled.transpose(1, 2)
    return upsampled

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class RFCXNet(nn.Module):
    def __init__(self, model_name):
        super(RFCXNet, self).__init__()
        self.model_name = model_name
        self.n_label = N_LABEL

        base_model = timm.create_model(model_name, pretrained=True)
        h_idx, n_dense = MODEL_HEADER_INFO[model_name]        

        # 過去学習に使ったモデルをロードするためヘッダーの名前を変える
        if self.model_name in ["resnet18", "efficientnet_b0"]:
            self.resnet_head = nn.Sequential(*list(base_model.children())[:h_idx])
        elif self.model_name == "resnest50d":
            self.resnest50d_head = nn.Sequential(*list(base_model.children())[:h_idx])
        else:
            self.model_head = nn.Sequential(*list(base_model.children())[:h_idx])
               
        # CBAM用
        #self.ca = ChannelAttention(n_dense)
        #self.sa = SpatialAttention()

        self.fc_a = nn.Conv1d(n_dense, self.n_label, 1, bias=False)
        self.fc_b = nn.Conv1d(n_dense, self.n_label, 1, bias=False)

    def forward(self, x, perm=None, gamma=None):  # input x: (batch, channel, Hz, time)
        frames_num = x.shape[3]
        x = x.transpose(3, 2)  # (batch, channel, time, Hz)

        # (batch, unit, time, Hz)
        if self.model_name in ["resnet18", "efficientnet_b0"]:
            h = self.resnet_head(x)  
        elif self.model_name == "resnest50d":
            h = self.resnest50d_head(x)
        else:
            h = self.model_head(x)
        
        # CBAM用
        #h = self.ca(h) * h
        #h = self.sa(h) * h

        if perm is not None:
            h = gamma * h + (1 - gamma) * h[perm]
    
        h = F.relu(h)
        ti_pool = torch.mean(h, dim=3)  # (batch, unit, time)

        xa = self.fc_a(ti_pool)  # (batch, n_class, time)
        xb = self.fc_b(ti_pool)  # (batch, n_class, time)
        xb = torch.softmax(xb, dim=2)

        # time pool
        clipwise_preds_att_ti = torch.sum(xa * xb, dim=2)
        segmentwise_output_ti = interpolate(xa, 32)

        return {
            "clipwise_preds_att_ti": clipwise_preds_att_ti,
            "segmentwise_output_ti": segmentwise_output_ti,
        }
```
