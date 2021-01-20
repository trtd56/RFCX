# my_solution

# memo

## ToDo

### Backlog

- Pseudo labeling
  - testデータ
  - 外部データ
  - trainの他のラベル
  - fpデータ
- TTA
- SWA
- Model選定
  - ViT
  - CBAM
- songtype_idの考慮

### Doing

- 一旦普通に学習してから、ラベルが付いているもののみで勾配計算する2st制
  - 結果(CV)
    - 500 Sampling: CV=0.7683 / LB=0.868
    - Sampling無し: CV=0.7766 / LB=0.874
    - pseudo seq: CV=0.7766 /LB=
    - mix up: CV=0.8030 /LB=
    - mix up(bugfix): CV=0.8053 /LB=0.846
    - mix up only positive: CV=0.7948 / LB=0.842
    - last_mixup: CV= /LB=
    - pseudo seq(0.9): CV= /LB=

### Done

- 明確にラベルがつけられているもの以外の勾配を計算しないようにして学習
  - 結果(Single Fold)
    - 従来: Local 1 fold=0.7845 / LB=0.731
    - 今回: Local 1 fold=0.7278 / LB=0.778
    - サンプリング: Local 1 fold=0.6844 / LB=0.772
    - 2nd stage(+ 5 epoch): Local 1 fold=0.7661 / LB=0.808

## 後で見る(まとめ系記事やNotebook)

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
|exp0107|very bad|X|not_bright_and_gamma, lr=3e-3|
|exp0108|0.8219|0.798|not_bright_and_gamma, lr=3e-4|
|exp0109|0.7889|0.842|not_bright_and_gamma, only_clip_loss|
|exp0110|0.8184|0.831|not_bright_and_gamma, clip_with_max|
|exp0111|0.7959|0.828|not_bright_and_gamma, clip_with_pseudo|
|exp0112|very bad|X|not_bright_and_gamma, clip_with_seq_nega0|
|exp0113|0.7608|0.793|not_bright_and_gamma, clip_ave_max_pool_sum|
|exp0114|0.7843|0.830|not_bright_and_gamma, bias=True|
|exp0115|0.7888||not_bright_and_gamma, clamp_att|
|exp0116|very bad|X|not_bright_and_gamma, clip_ega_0|
|exp0117|0.7825|0.858|not_bright_and_gamma, only_clip_loss, resnest|
