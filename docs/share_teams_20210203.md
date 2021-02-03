# 2021/02/03

## mixup last layer

- きっかけは[この記事](https://akichan-f.medium.com/%E6%9C%80%E7%B5%82%E5%B1%A4%E3%81%A7mixup%E3%81%97%E3%81%9F%E3%82%89%E8%89%AF%E3%81%95%E3%81%92%E3%81%A0%E3%81%A3%E3%81%9F%E4%BB%B6-bd2ff167c388)
- mixupすると特徴空間が重なってしまう恐れがある
- 特にスペクトログラムのような特徴量が局所的に現れる画像は悪影響が大きそう
- また今回のコンペのデータはかなりノイジーなので、画像データではっきり混ぜるより、NNの分散表現のなかでゆるっと混ぜたほうが良さそうだと思った


### 実装

#### mixup関数
- モデルの中で使うので、gammaとpermを外から与える
```python
def mixup(input, gamma, perm):
    perm_input = input[perm]
    return input.mul_(gamma).add_(1 - gamma, perm_input)
```

### model

```python
class RFCXNet(nn.Module):
    def __init__(self, model_name):
        super(RFCXNet, self).__init__()
        base_model = timm.create_model(model_name, pretrained=True)
        self.model_head = nn.Sequential(*list(base_model.children())[:-2])
          .
          .
          .

    def forward(self, x, perm, gamma):
        frames_num = x.shape[3]  # input x: (batch, channel, Hz, time)
        x = x.transpose(3, 2)  # (batch, channel, time, Hz)
        h = self.model_head(x)  # (batch, unit, time, Hz)
        
        # ここでmixup
        h = gamma * h + (1 - gamma) * h[perm]
            
        # これはあくまで一例
        # max poolingとかdense層に接続するために色々な方法が使える
        h = F.relu(h)
        ti_pool = torch.mean(h, dim=3)  # (batch, unit, time)
          .
          .
          .
```

#### 学習ループ内

```python
          .
          .
          .
        # mixupのパラメータ作成
        gamma = beta(0.1, 0.1)
        perm = torch.randperm(X.size(0)) 
        
        # 目的変数のmixup
        y_mixup = mixup(y, gamma, perm)
        
        # モデルのfoword+mixup
        y_pred = model(X, perm, gamma)
        
        loss = loss_fn(y_pred, y_mixup)
          .
          .
          .
```

### 実験結果

#### 旧実験(Precisionは測ってない)
学習方法は1st stage

- 学習の中でpseudo labelingしてるらへん

|mixup type|CV|LB|
|--|--|--|
|従来|0.8399|0.831|
|last layer|0.8391|0.805|

- negativeデータをサンプリングして追加してるらへん

|mixup type|CV|LB|
|--|--|--|
|従来|0.8352|0.833|
|last layer|0.8287|0.828|

- 画像の分割方法を工夫してるらへん

|mixup type|CV|LB|
|--|--|--|
|従来|0.8347|0.792|
|last layer|0.8368|0.792|

#### pseudo0.9 + Focal

|mixup type|CV|Precision|LB|
|--|--|--|--|
|従来|0.9392|0.4406|0.912|
|last layer|0.9434|0.4375|0.907|

#### pseudo0.5 + Focal + positive重み減衰

|mixup type|CV|Precision|LB|
|--|--|--|--|
|従来|0.9663|0.3636|0.914|
|last layer|0.9589|0.657|0.942|


### 考察

- 基本的にうまく行ってなかったが、`画像の分割方法を工夫してるらへん`からCVが高くなることを考えると、単純に学習能力を向上させているっぽい
    - それまでは画像がrandom cropだったので、なにもないところをラベル付きで学習したりしてたのかも
- `pseudo0.5 + Focal + positive重み減衰`でぐぐっとLBも上がったのは、隠れているラベルをpseudoで見つけられたから？
- 実装は簡単なので、ダメ押し的に使うといいかも
    - CVではLWLRAPをキープしてPrecisionを上げる
