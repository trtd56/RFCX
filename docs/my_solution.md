# my_solution
- Model: ResNet18
- 5 fold CV

# memo

## 課題
- testとlocalが違う
- trainにノイズが多い
- 明確なネガティブ以外でnagative lossを学習しないようにする

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
- [x] [鳥コンペの上位ソリューション](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/197873)
- [x] [評価スコア計算のPytorch実装](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/198418)
- [x] [切り取ったデータセット](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/199025)

## 実験

|実験名|CV|LB|memo|
|--|--|--|--|
|exp0001|0.77662|0.734|baseline P100|
|exp0002|0.79004|0.723|CosAnealScheduler|
|exp0003|0.77224|0.698|lossのpseudo無し|
|exp0004|0.77802|0.713|FreqMask|
|||||
|exp0005|0.7792|0.692|baseline(T4)|
|exp0006|0.7869|0.710|gamma noise|
|exp0007|not good|X|softmax→tanh|
|exp0008|not good|X|softmax→sigmoid|
|exp0009|0.7879|0.731|pos_weight|
|exp0010|0.7813|0.729|random brightness→[albumentations](https://github.com/albumentations-team/albumentations)|
|exp0011|0.7875|0.714|gamma before norm|
|exp0012|0.7859|0.708|step LR Scheduler|
|||||
|exp0013|0.8223|0.769|Augument全部のせ+pos weigth+CosAnealScheduler|
|exp0014|not good|X|fpデータの学習(FPが学習できるか)|
|exp0015|0.803|0.762|mixupのOFF|
|exp0016|0.7955|0.730|softmax+tanh|
|exp0017|0.8136|0.762|AdamW|
|exp0018|0.8171|0.761|Focal Loss|
|exp0019|0.8144|0.780|framewise_outputのしきい値をへらす|
|exp0020|not good|X|att poolとmax poolのlossの和|
|||||
|exp0021|0.8143|0.828|cutしてdetection|
|exp0021|0.8143|0.814|cutしてdetection max|
|exp0022|0.8154|0.816|framewise_outputのしきい値0.3|
|exp0023|0.8106|0.805|mixupのalpha=0.2|
|exp0024|0.8159|0.804|mixup last layer|
|exp0025|0.8095|0.814|window size 256|
|exp0025|0.8095|0.799|window size 256 max|
|exp0025|X|0.835|mix 256, 512|
|exp0026|0.7935|0.793|lr 3e-4|
|exp0027|0.8089|0.772|lr 3e-4, Label Smoothing|
|exp0028|0.833|0.824|att poolとmax pool, 1:0.5|
|exp0029|0.8308|0.813|att poolとmax pool, 1:1|
|exp0030|0.8274|0.820|att poolとmax pool + mixup last layer + framewise_outputのしきい値0.3|
|exp0031|0.8332|0.803|att poolとmax pool + pos weight loss|
|exp0032|0.8293|0.785|att poolとmax pool + pos weight loss + mixup last layer|
|exp0033|0.792|0.807|low pass filtering|
|exp0034|0.8259|0.801|fpデータの学習(学習データにのみ追加)|
|||||
|exp0035|0.839|0.826|fpデータの学習(学習データにのみ追加), att poolとmax pool + pos weight loss|
|exp0036|0.8363|0.805|framewise_outputのしきい値0.3|
|||||
|exp0036|0.8399|0.831|framewise_outputのしきい値0.7|
|exp0037|0.8358|0.815|メインloss 0.5|
|exp0038|0.8358|0.783|Label Smoothing|
|exp0039|0.8377|0.806|positiveのlossを削除|
|exp0040|0.8391|0.805|with last mixup|
|exp0041|not good|X|RandomCrop|
|exp0042|0.8287|0.837|segmentwise_output|
|exp0043|0.8337|0.829|xa * xb|
|||||
|exp0044|0.8315|0.830|fix imprement|
|exp0045|0.8286|0.803|Hz pred loss|
|exp0046|0.828|0.811|fix mixup|
|exp0047|0.8324|0.821|pseudo 0.5|
|||||
|exp0048|0.8304|0.799|nega label fix|
|exp0049|0.8339|0.791|メインをFocal loss|
|exp0049|0.8285|0.828|メインをFocal loss+weight|
|exp0050|0.8341|0.802|メインをFocal loss+weight * 1/2|
|exp0051|0.8352|0.833|negativeデータ減らす→30|
|exp0052|0.8246|0.823|negativeデータ減らす→10|
|exp0053|0.8241|0.817|Label Smoothing Clip|
|exp0054|0.8329|0.825|Label Smoothing Seq|
|exp0055|0.8287|0.828|with last mixup|
|exp0056|0.8324|0.821|with last mixup, 0.5|
|exp0057|0.8305|0.797|negativeのlossを削除 BCE|
|exp0058|0.8414|0.810|negativeのlossを削除 Seq|
|exp0059|0.838|0.816|Window768|
|exp0060|0.8085|X|max128|
|exp0061|0.8303|0.826|min64|
|exp0062|not good|X|each32|
|exp0063|0.8389|0.774|negativeのlossを削除 Seq, not 0.5|
|exp0063|0.8424|0.766|negativeのlossを削除 Seq, max Focal 0.5|
|exp0064|||negativeのlossを削除 Seq, negative smoothinng|
|exp0064|||negativeのlossを削除 Seq, pseudo 0.9|
|||||
|exp0045|||OOF noisy label対策|
|||||
|exp0039|||fpデータのpseudo|
|||||
|exp0039|||CBAM|
|||||
|exp0025|||Densenet|
|exp0025|||Pseudo labeling|
|exp0025|||TTA|

#### best loss
- exp0013: loss=0.694, LB=0.677
