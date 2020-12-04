# my_solution
- Model: ResNet18
- 5 fold CV

# memo

## やりたいこと
- [ ] Adversarial
  - [参考Notebook](https://www.kaggle.com/tunguz/adversarial-rainforest)

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

|実験名|Fold|local score|LB|memo|
|--|--|--|--|--|
|exp0001|0|0.7715|0.695|baseline|
|exp0001|1|0.761|0.623|baseline|
|exp0001|2|0.7775|0.596|baseline|
|exp0001|3|0.7869|0.625|baseline|
|exp0001|4|0.7862|0.627|baseline|
|exp0001|CV|0.77662||baseline before sigmoid|
|exp0001|CV|0.77662||baseline after sigmoid|
||||||
|exp0002||||CosAnealScheduler|
||||||
|exp0003||||lossの割合1:1:1に|
||||||
|exp0004||||lossのpseudo無し|
||||||
|exp000５||||fpデータの利用|
||||||


