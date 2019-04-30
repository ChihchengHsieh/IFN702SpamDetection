# IFN702SpamDetection

To solve the problem of gradient vanish:

I have changed:
```python
out = self.rnn(out)[0][:, -1, :]         
```
to 
```python
out = self.rnn(out)[0].sum(dim=1)
```

## Best tunned results 

# SSCL
0.9384
![](https://github.com/ChihchengHsieh/IFN702SpamDetection/blob/master/TrainingResults/SSCL_1.png?raw=true)

# GatedCNN
0.9476
![](https://github.com/ChihchengHsieh/IFN702SpamDetection/blob/master/TrainingResults/Gated_1.png?raw=true)

# SelfAttn (Transformer)
0.9345
![](https://github.com/ChihchengHsieh/IFN702SpamDetection/blob/master/TrainingResults/SelfAttn_3.png?raw=true)
