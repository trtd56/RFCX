# 2021/02/08

## 提出ファイルの共有



|model|lwlap|preci|recall|LB|memo|Drive|
|--|--|--|--|--|--|--|
|resnet18|0.9589|0.657||0.942|現状SingleでBESTだが、出力値が低い|[csv](https://drive.google.com/file/d/1J_nHAgEpVaZnw-USU9OS_U5VOcZsBHkB/view?usp=sharing)|
|resnet18|0.9693|0.9590|0.7336|0.936|現状ベストプラクティス★|[csv](https://drive.google.com/file/d/1gy2MmNY5OyDzx-sTe5ldJPkk7RBss_9s/view?usp=sharing)|
|resnet18|0.9457|0.5744|0.08893|0.921|windowサイズ256だが、勾配累積を使っていないので出力値が低い|[csv](https://drive.google.com/file/d/1Q5wOeESQp0pc8up6lzaukv9UGC7dBP1G/view?usp=sharing)|
|resnet18|0.9657|0.9665|0.7125||CBAM|[csv](https://drive.google.com/file/d/106dAE2HwxjPnZL4GGCRCn6lOsUjqcqoX/view?usp=sharing)|
|resnest50|0.9397|0.8985||0.935|現状ベストプラクティス★|[csv](https://drive.google.com/file/d/1rI3vPWG8GxsR4L9S_lZLaDLiLivIqX9g/view?usp=sharing)|
|densener121|0.9392|0.8706||0.926|現状ベストプラクティス★|[csv](https://drive.google.com/file/d/1sTcQg3ZneF4Tjp4JGWknCgULq6KnzvZE/view?usp=sharing)|
|efficientnet_b0|0.8897|0.8748||0.922|現状ベストプラクティス★|[csv](https://drive.google.com/file/d/15qDjo-82ffp8jtQMXKTt6nFk4oJpqtUw/view?usp=sharing)|
|efficientnet_b0|0.9002|0.8758|||↑のパラメータ調整|[csv](https://drive.google.com/file/d/1CbtU3kaiFjoAio-agM1txg0CZHvf9WpC/view?usp=sharing)|

- submitしていなかったり、実装が間に合わずスコアを計算していないものもある
- precision/recallは閾値0.5で計算している
- 現状ベストプラクティス
  - 明確にラベルが付いていないlossを計算しないのとpseudo labelingのcycle
  - 勾配累積
  - mixup lastlayer
  - positive lossをバッチの中で重み減衰
- アンサンブルは★がついている4個の単純平均でLB=0.941
