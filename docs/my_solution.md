# my_solution

# memo

## ToDo

### Backlog


- Model選定
  - ViT
  - CBAM
- songtype_idの考慮
- Pseudo labeling
  - testデータをつかう
  - その他の外部データを使う
- 一般的な改善手法
  - mixup
  - last_mixup
  - TTA
  - SWA
  - Focal Loss([CPMPのやつ](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075)が良さそう)

### Doing

- pseudo labeling
  - train(tp/fp)の他のラベル: CV=0.9260 / LB=0.903
  - frame wiseでpseudo labelng: CV=0.7777 / LB=0.874
  - negativeもpseudo labeling(0.1未満とかの閾値)
- denoise
  - 2st stage: CV=0.7579 / LB=0.828
  - 3rd stage: CV= / LB=
### Done
    
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
