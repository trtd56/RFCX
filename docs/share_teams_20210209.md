# 2021/02/09

## Share about split the audio

I train by google colab using [Theo Viel's npz dataset](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/198048)(32 kHz, 128 mels), because I don't have GPU machine and enough monery to rent Cloud machine.
The dataset can be regarded as 128x3751 size image.


Cut out the entire range of audio data by covering it little by little


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
