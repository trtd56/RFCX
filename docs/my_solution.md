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

### モデル選定

|実験名|CV|LB|memo|
|--|--|--|--|
|exp0100|0.8282|0.793|Resnet18 max|
|||0.816|Resnet18 avg|
|exp0101|0.8313|0.803|DenseNet max|
|||0.825|DenseNet avg (以降基本avg)|
|exp0102|0.8202|0.833|EfficientNets-B0|
|exp0103|0.8397|0.851|ResNests50|
|exp0104|0.8175|0.799|MobileNet-V2|

そもそもベースの精度が低そう

### モデル選定

ResNet18で実験(Trust LB)

|実験名|CV|LB|memo|
|--|--|--|--|
|exp0105|0.8279|0.808|not_bright_and_gamma|
|exp0106|0.8053|0.804|not_bright_and_gamma, not_fp30|
