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

We can change this N_SPLIT_IMG, WINDOW, COVER. I tried  N_SPLIT_IMG=16, WINDOW=256, and COVER=23 but not good.


|WINDOW|model|lwlap|precision|recall|LB|memo|
|--|--|--|--|--|--|--|
|512|resnet18|0.9693|0.9590|0.7336|0.936|my best single model|
|256|resnet18|0.9457|0.5744|0.08893|0.921|↑same method other than split size|


soft frame wise prediction.

---
I use 

1st stage
- I think it is pre-train.
- detect sound exist point from t_max and t_min, and cut image (3, 128, 512)
- use original label
- sampling 30 data from fp_train on each epochs(they are np.zero(24))

2nd stage
- load the model weight which was trained 1st stage
- use all data(tp_train, fp_train)
- cut out the entire range of audio data by covering it little by little
- calculate loss only those labeled in tp_train and fp_train

3rd stage
- same with 2nd stage
- re-labeling by 2nd stage model

4th stage
- same with 2nd stage
- re-labeling by 3rd stage model

I tried the 5th stage(re-labeling by the 4th stage model), but the result was not good.

Share about split the audio
