# 2021/01/22

## 現状BEST手法の共有
共通設定
- ベースモデル: ResNet18
  - SEDモデル
- 学習方法: スペクトログラムを512毎にcropして各予測のmax ploolingをとる
- CV: MultilabelStratifiedKFold
- clip wiseで学習してsegment　wiseで予測
- ハイパーパラメータ
  - Adam
    - learning_rate = 1e-3
  - CosineAnnealingLR
    - _max = 10
- augumentation
  - spec aug(time, Hz)

### 1st stage
- 普通に学習する
- FPデータをepoch毎に30件ずつサンプリングして学習
- ハイパーパラメータ
  - epoch = 50
  - batch size = 64
- augumentation
  - mixup
- スコア: CV=0.7889 / LB=0.842

### 2nd stage
- 1st stageのモデルの重みからfine-tuning  
- 明確にtp_train, fp_trainでラベル付けされているもの以外の勾配を計算しない
- ハイパーパラメータ
  - epoch = ５
  - batch size = ３２
- スコア: CV=0.7766 / LB=0.874

### 3rd stage
- 1st stageのモデルの重みからfine-tuning  
- 2nd stageの最終モデルでpseudo labeling
  - 閾値0.9で5foldのモデルのうち、3件以上がTrueと予測したラベルをclip wiseで付与
- ハイパーパラメータ
  - epoch = 5
  - batch size = 32
- スコア: CV=0.9260 / LB=0.903
