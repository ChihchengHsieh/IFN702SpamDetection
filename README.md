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
![](https://github.com/ChihchengHsieh/IFN702SpamDetection/blob/master/TrainingResults/SSCL_1.png?raw=true)

# GatedCNN
![](https://github.com/ChihchengHsieh/IFN702SpamDetection/blob/master/TrainingResults/Gated_1.png?raw=true)

# SelfAttn (Transformer)
![](https://github.com/ChihchengHsieh/IFN702SpamDetection/blob/master/TrainingResults/SelfAttn_3.png?raw=true)
