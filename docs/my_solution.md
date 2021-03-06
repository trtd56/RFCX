# my_solution

# memo

## ToDo

### Backlog

- cut imageを1st
- その他の外部データのPseudoLabeling
- 一般的な改善手法
  - TTA
  - SWA
- ViT

### Doing

|exp|CV|Prec|Recall|LB|memo|
|--|--|--|--|--|--|
|184|0.936|0.9529|0.7057|0.936|smoothing=0.45|
|185|X|X|X|X|label weight|
|186|0.972|0.9118|0.8682||pos w 2|
|186|0.928|0.8387|0.9382|0.928|pos w 4|
|187|0.969|0.9527|0.722||add gamma noise|

#### アンサンブル
順当に行く場合
- reset18, densenet121, efficientnet_b0, resnest50: 0.936
- reset18, densenet121, efficientnet_b0(ベース強化), resnest50
- reset18, densenet121, efficientnet_b0(ベース強化), resnest50, きょうへいさんのモデル
- stacking


### Done

- 3rd stage (pseudoはpositive onlyが良さそう)
  - Focal Loss(weightはpositiveのみ):  CV=0.8952 / LB=0.911
  - Focal Loss(weightはnegativeも): CV=0.8936 / LB=0.900
  - tp_trainのみpseudo: CV=0.8999 / LB=0.840
  - Label Weight: CV=0.925 / LB=0.869
  - mixup: CV=0.9346 / LB=0.912
  - mixup layer: CV=0.9392 / LB=0.907
  - mixup negatveも: CV=0.9062 /LB=0.907
- testデータのpseudo: CV=0.9325 / LB=0.909
- pseudo labeling
  - train(tp/fp)の他のラベル: CV=0.9260 / LB=0.903
  - frame wiseでpseudo labelng: CV=0.7777 / LB=0.874
  - negativeもpseudo labeling(0.1未満とかの閾値)
    - exp0138: CV=0.8513 / LB=0.850
    - exp0141(bug fix): CV=0.9446 / LB=0.878
- denoise
  - 2st stage: CV=0.7579 / LB=0.828
  - testも同じ処理: LB=0.857
  - 3rd stage: CV=0.8954 / LB=0.858
- testにtrainから抽出したノイズを乗せる→[データ作成](https://www.kaggle.com/takamichitoda/rfcx-add-noise-to-test?scriptVersionId=52907736), [Dataset](https://www.kaggle.com/takamichitoda/rfxc-add-noise-test-data), [推論Notebook](https://www.kaggle.com/takamichitoda/rfxc-add-noise?scriptVersionId=52980971)
  - Origin: CV=0.9346 / LB=0.912
    - ノイズ除去のやつで抽出: LB=0.902
    - Bestモデルで予測(x10): LB=0.767
- ~~Model選定~~
  - ~~ViT~~
  - ~~CBAM~~
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

### focal_mixup_pseudoの改善

|exp|CV|Prec|LB|memo|
|--|--|--|--|--|
|150|0.9131|0.3565|0.890|Under sampling|
|151|0.9412|0.6282|0.868|Label Weight min|
|152|0.9487|0.6270|0.874|exp0151で閾値=0.75|
|153|0.9507|0.6417|0.876|exp0151で閾値=0.5|
|154|0.9538|0.4227|0.923|exp0146で閾値=0.5|
|160|0.9625|0.3868|0.918|pseudo thr=0.3|

### songtypeの考慮

|exp|CV|Prec|LB|memo|
|--|--|--|--|--|
|154|0.5743(0.8507)|0.3244(0.5288)||songtype_idの考慮1st|
|154|0.6841(0.9136)|0.2151(0.2824)||songtype_idの考慮2nd|
|154|0.7848(0.8745)|0.3124(0.3131)|0.826|songtype_idの考慮3rd|
|155|0.7558|||songtype_idの考慮2nd, 計算修正|
|156|0.7958|0.4475|0.806|songtype_idの考慮1st, 計算修正|
|156|0.7611|0.3480||songtype_idの考慮2nd, 計算修正|
|156|0.9437|0.3393|0.900|songtype_idの考慮3rd, 計算修正|
|156|0.8516|0.4007|0.825|songtype_idの考慮4th, 計算修正|
|156|0.9560|0.3963|0.829|songtype_idの考慮5th, 計算修正|


### アンサンブルの種づくり
|exp|CV|Prec|LB|memo|
|--|--|--|--|--|
|157|0.7862|0.3907||densenet121, 1st|
|157|0.7645|0.2856||densenet121, 2nd|
|158|0.7768|0.4096||efficientnet_b0, 1st|
|158|0.7567|0.3217||efficientnet_b0, 2nd|
|158|0.8964|0.2153|0.857|efficientnet_b0, 3rd|
|164|0.7621|0.315||efficientnet_b0, 2nd, 8 epoch|
|164|0.8753|0.8768||efficientnet_b0, 3rd, use resnet pseudo|
|166|0.9392|0.8706|0.926|densenet121, 3rd, use resnet pseudo|
|168|0.8897|0.8748|0.922|efficientnet_b0 mixup_last|
|170|not good|not good|X|MobileNet|
|177|X|X|X|densenet121, lr=5e-4|
|178|0.9002|0.8758||efficientnet_b0, lr=5e-4|

### windowサイズの変更
|exp|CV|Prec|LB|memo|
|--|--|--|--|--|
|183|0.8063|0.3094||2nd, w256|
|183|0.964|0.3418|0.919|3rd, w256|
|183|0.9457|0.5744|0.921|4th, w256|

### モデルのベース改善

|exp|CV|Prec|LB|memo|
|--|--|--|--|--|
|161|0.9691|0.427|0.922|勾配累積|
|162|0.9663|0.3636|0.914|Cycle Pseudo|
|163|0.9590|0.5667|0.937|pos loss weight|
|165|not good|not good|X|pos loss weight, alpha_min|
|167|0.9589|0.657|0.942|pos loss weight, mixup_last|
|169|0.9397|0.8985|0.935|exp0117の重みを使ってResnest|
|180|0.9693|0.9590|0.936|resnet18, 勾配累積の利用|
|181|0.9451|0.7991|0.921|resnet18, cycle3|
|182|0.9657|0.9665||resnet18, CBAM|
