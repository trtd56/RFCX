# my_solution

# memo

## ToDo

- CVとLBの相関をとる
  - noisy labelっぽいのでOOFでラベルの付け直し
  - Trust LBがよいかも?
- Pseudo labeling
  - testデータ
  - 外部データ
  - trainの他のラベル
  - fpデータ
- TTA
- Model選定
  - ViT
  - CBAM
- songtype_idの考慮

## 後で見る(まとめ系記事やNotebook)
- 

## 実験

### LB/CVの相関確認

これでダメそうならTrust LBか？

|実験名|CV|CV max|LB|LB max|memo|
|--|--|--|--|--|--|
|exp0065|0.8347|0.8287|0.792|0.789|Focal|
|exp0065|0.8347|0.8287|0.792|0.794|Focal max|
|exp0067|0.8379|0.8340|0.802|0.803|Base|
|exp0067|0.8379|0.8340|0.802|0.798|Base max|
|exp0068|0.8257|0.8246|0.790|0.785|not nega loss del|
|exp0073|0.8368|0.8324|0.792|0.793|with last mixup|

### モデル選定

|実験名|CV|LB|memo|
|--|--|--|--|
|exp0100|0.8282|0.793|Resnet18 max|
|||0.816|Resnet18 avg|
|exp0101|0.8313|0.803|DenseNet max|
|||0.825|DenseNet avg (以降基本avg)|
|exp0102|||EfficientNets-B0|
|exp0103|||ResNests50|
|exp0104|||MobileNet-V2|

|exp0103|||mobilenet|
