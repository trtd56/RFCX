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
|exp0001|CV|0.77662|0.734|baseline before sigmoid|
|exp0001|CV|0.77662|0.711|baseline after sigmoid|
||||||
|exp0002|0|0.7649|0.696|CosAnealScheduler|
|exp0002|1|0.7929|0.626|CosAnealScheduler|
|exp0002|2|0.7921|0.652|CosAnealScheduler|
|exp0002|3|0.7942|0.650|CosAnealScheduler|
|exp0002|4|0.8061|0.600|CosAnealScheduler|
|exp0002|CV|0.79004|0.723|CosAnealScheduler before sigmoid|
|exp0002|CV|0.79004|0.711|CosAnealScheduler after sigmoid|
||||||
|exp0003|0|0.78||lossのpseudo無し|
|exp0003|1|0.7642||lossのpseudo無し|
|exp0003|2|0.7742||lossのpseudo無し|
|exp0003|3|0.7714||lossのpseudo無し|
|exp0003|4|0.7714||lossのpseudo無し|
|exp0003|CV|||lossのpseudo無し before sigmoid|
|exp0003|CV|0.77224|0.698|lossのpseudo無し after sigmoid|
||||||
|exp0004||||FreqMask|
||||||
|exp0005||||gamma noise|
||||||
|exp0006||||baseline(T4)|
||||||
|exp0007||||freq Attention|
||||||
|exp0008||||fpデータの利用|
||||||

