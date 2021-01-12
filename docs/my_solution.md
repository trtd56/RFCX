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
|exp0102|||EfficientNets-B0|
|exp0103|||ResNests50|
|exp0104|||MobileNet-V2|
