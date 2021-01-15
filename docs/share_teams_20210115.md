# 2021/01/15

## 現状までの共有

### 課題
**なんといってもCV/LBの相関が取れないこと！！！！**
- [年始に発覚したバグを使った人によるとPublicとPrivateは結構相関してる](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/207901#1134198)みたいなので、Trust LBでもよいかも？
- 前回鳥コンペのアライさんのようにラベルをつけなおすのも考えられると思いますが、[shinmura0さんは効かなかったみたい](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/209684)

### モデル

ベースはResNet18
- [前回鳥コンペの1stの人はDenseNet使ってた](https://www.kaggle.com/c/birdsong-recognition/discussion/183208)
- densenetの他にefficientnet, mobilenet, resnestsを試した
  - resnestsがよさげ(CV0.8397/LB0.851)
  - [shinmura0さんによるとEfficientNetがいいそう](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/209041)
- 実験を早く回したいので基本ResNet18のままで実験してる

ベースを前回鳥コンペのアライさんみたいにSEDにする形にする
- [前回私が試した疑似ラベリングを使っている](https://www.kaggle.com/c/birdsong-recognition/discussion/183196)
  - これが良かったわけではなく、前回のコードを流用しているだけ
- segmentwise_outputのmaxを予測につかっている(from [shinmura0さんDiscussion](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/209684))

#### パラメータ

- batch_size = 64 
- num_epochs = 50
- mixup alpha = 0.1
- Adam
  - learning_rate = 1e-3
- CosineAnnealingLR
  - t_max = 10

[スケジューラーが大事というご意見も](https://www.kaggle.com/kneroma/inference-tpu-rfcx-audio-detection-fast)

### データ読み込み
- 計算資源がしょぼいので[計算済みのスペクトログラムのnpz](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/198048)をcolabでDLして利用
- 学習時はt_maxとt_minから実際に鳴いている声があると思われる部分を切り取る
  - サイズは512で、t_maxとt_minの中心がランダムに32, 64, 128, 256に移動するように切り取る
- 推論時は学習時と同じサイズで予測するため、対象を512ずつ切り取り、各予測のmax_poolingをとる
- fpデータをepoch毎に30件ずつサンプリングしてラベル0で学習データに追加している

#### augumentation
効いたやつ
- spec aug
- mix up
  - 中間層でmix upする手法(Manifold mixup)も試したけど良かったり悪かったりでよくわからない

効いたり効かなかったり
- ガンマノイズ
- 光調整

効かない
- cutout

TTAは絶対効くだろうと思ってるけど試してない

### CV

MultilabelStratifiedKFold
- ファイル毎でsplit
  - ファイルによってはマルチラベル問題になる
  
### 損失

4つのlossの荷重和
① : clip wiseのattention出力  
② : clip wiseのmax pooling出力  
③ : seq wiseの疑似ラベル  
④ : seq wiseの出力

`loss = ① + ②*0.5 + ③ + ④*0.5`

効いたり効かなかったり
- Focal Loss
- ノイジーなデータらしい(鳴いている鳥がラベル付けされてなかったりする)が、tpラベルはわりと真っぽいので、negativeのlossに0がけしてpositive lossだけ計算する
  - clip wiseのlossは駄目
  - seq wiseのlossは少し効いた

効かない
 - Label Smoothing

### アンサンブル
- 5foldの平均
- 複数モデルのstackingやaveragngはまだ試してない
  - 高スコア公開Notebookみると上がるのは目に見えてるが
    - モデルのバリエーションが大事になりそう
    - ベースのモデルのスコアを上げることを優先してた


