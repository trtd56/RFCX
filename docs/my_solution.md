# my_solution
- Model: ResNet18
- 5 fold CV

# memo

## やりたいこと
- [ ] Adversarial
  - [参考Notebook](https://www.kaggle.com/tunguz/adversarial-rainforest)
- [ ] TTA
- [ ] Pseudo Labeling

## Dataset
### songtype_idについて
- 1と4の値をとる
- 主催者が一般的だと思ったラベルが1でそれ以外の少数ラベルが4
- 参考Discussion: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/197735

### スペクトラムを画像に
- 計算量の削減
- Discussion: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/198048

## 後で見る(まとめ系記事やNotebook)
- [ ] [Previous Audio Competitions](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/197737)
- [ ] [Frogs and birds sounds in nature: research, resources and more](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/197751)
- [x] [COLA: 音声の汎用pre-trainモデル](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/197805)
- [x] [アライさんの音声のData AugumentationのNotebook](https://www.kaggle.com/hidehisaarai1213/rfcx-audio-data-augmentation-japanese-english)
- [x] [pytorchの推論](https://www.kaggle.com/kneroma/inference-resnest-rfcx-audio-detection)
- [ ] [鳥コンペの上位ソリューション](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/197873)
- [x] [評価スコア計算のPytorch実装](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/198418)
- [x] [切り取ったデータセット](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/199025)

## 実験

|実験名|CV|LB|memo|
|--|--|--|--|
|exp0001|0.77662|0.734|baseline P100|
|exp0002|0.79004|0.723|CosAnealScheduler|
|exp0003|0.77224|0.698|lossのpseudo無し|
|exp0004|0.77802|0.713|FreqMask|
|exp0005|0.7792|0.692|baseline(T4)|
|exp0006|0.7869|0.710|gamma noise|
|exp0007|not good||softmax→tanh|
|exp0008|not good||softmax→sigmoid|
|exp0009|0.7879|0.731|pos_weight|
|exp0010|0.7813|0.729|random brightness→[albumentations](https://github.com/albumentations-team/albumentations)|
|exp0011|0.7875|0.714|gamma before norm|
|exp0012|0.7859|0.708|step LR Scheduler|
|exp0013|0.8223|0.769|Augument全部のせ+pos weigth+CosAnealScheduler|
|exp0014|||fpデータの利用|
|exp0015|||mixupのOFF|
|exp0016|||mixupのalpha=0.2|
|exp0017|||Focal Loss|
|exp0018|||Label Smoothing|

#### best loss
- exp0013: loss=0.694, LB=0.677
