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

# SSCL
![](https://github.com/ChihchengHsieh/IFN702SpamDetection/blob/master/SSCL.png?raw=true)

# GatedCNN
![](https://github.com/ChihchengHsieh/IFN702SpamDetection/blob/master/GatedCNN.png?raw=true)

# SelfAttn (Transformer)
![](https://github.com/ChihchengHsieh/IFN702SpamDetection/blob/master/SelfAttn.png?raw=true)
