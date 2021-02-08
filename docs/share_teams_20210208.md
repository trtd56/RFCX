# 2021/02/08

## モデルと提出ファイルの共有



|model|lwlap|preci|recall|LB|memo|Drive|
|--|--|--|--|--|--|--|
|resnet18|||||現状SingleでBESTだが、出力値が低い||
|resnet18|||||現状ベストプラクティス||
|resnet18|||||windowサイズ256だが、勾配累積を使っていないので出力値が低い||
|resnest50|||||現状ベストプラクティス||
|densener121|||||現状ベストプラクティス||
|efficientnet_b0|||||現状ベストプラクティス||
|efficientnet_b0|||||↑のパラメータ調整||

現状ベストプラクティス
- 明確にラベルが付いていないlossを計算しないのとpseudo labelingのcycle
- 勾配累積
- mixup lastlayer
- positive lossをバッチの中で重み減衰
