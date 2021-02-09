# 2021/02/09

## Share about split the audio

I train by google colab using [Theo Viel's npz dataset](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/198048)(32 kHz, 128 mels), because I don't have GPU machine and enough monery to rent Cloud machine.
The dataset can be regarded as 128x3751 size image.

I set the window size to 512 and cut out the entire range of 60's audio data by covering it little by little.
Cover 49 pixels each, considering that important sounds are located at the boundaries of the division.
```python
N_SPLIT_IMG = 8
WINDOW = 512
COVER = 49

slide_img_pos = [[0, WINDOW]]
for idx in range(1, N_SPLIT_IMG):
    h, t = slide_img_pos[idx-1][0], slide_img_pos[idx-1][1]
    h = t - COVER
    t = h + WINDOW
    slide_img_pos.append([h, t])

print(slide_img_pos)
# [[0, 512], [463, 975], [926, 1438], [1389, 1901], [1852, 2364], [2315, 2827], [2778, 3290], [3241, 3753]]
```

I train and predict those images which cut by 128x512.

### training phase

Images were cut and padded to train by batch.

```python
def split_and_padding(X, y):
    x_lst = []
    y_lst = []
    for h, t in slide_img_pos:
        _X = X[:, :, :, h:t]
        _y = y[:, :, h:t]
        if _X.shape[3] != WINDOW:
            x_pad = torch.zeros(list(_X.shape[:-1]) + [WINDOW - _X.shape[3]])
            _X = torch.cat([_X, x_pad], axis=3)
            y_pad = torch.zeros(list(_y.shape[:-1]) + [WINDOW - _y.shape[2]])
            _y = torch.cat([_y, y_pad], axis=2)
        x_lst.append(_X)
        y_lst.append(_y)
    
    X = torch.cat(x_lst, axis=0)
    y = torch.cat(y_lst, axis=0)
    return X, y

for X, y in train_data_loader:
    # X shape is (2, 3, 128, 3751) batch_size, channel, image_height, image_width
    # y shape is (2, 24, 3751) batch_size, label, sequence
    X, y = split_and_padding(X, y)
    # X shape is (16, 3, 128, 512)
    # y shape is (16, 24, 512) 
    
    outouts = model(X)
    loss = loss_fn(outouts, y)
      .
      .
      .
```

### prediction phase

I predict each sliding window and extract max-pooling in the frame.
I call this soft frame-wise prediction.

```python
for X, _ in test_data_loader:
    preds = []
    for h, t in slide_img_pos:
        with torch.no_grad():
            outputs = model(X[:,:,:,h:t])
        _p, _ = outputs["segmentwise_output_ti"].sigmoid().max(2)
        preds.append(_p)
    test_pred, _  = torch.max(torch.stack(preds), dim=0)
      .
      .
      .
```

My model has two output, clipwise_output and framewise_output.
I use clipwise_output in training, and I use framewise_output in prediction.
This aproach came from [hinmura0's discussion thread](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/209684).


---

By the way, we can change this N_SPLIT_IMG, WINDOW, COVER. I tried  N_SPLIT_IMG=16, WINDOW=256, and COVER=23 but not good.

|WINDOW|model|lwlap|precision|recall|LB|memo|
|--|--|--|--|--|--|--|
|512|resnet18|0.9693|0.9590|0.7336|0.936|my best single model|
|256|resnet18|0.9457|0.5744|0.08893|0.921|↑same method other than split size|

## 2nd stage: cycle re-labeing





---





## A different approach from kudo



