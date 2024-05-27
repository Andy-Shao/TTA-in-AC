# Document info
https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5

# dataset link
https://urbansounddataset.weebly.com/urbansound8k.html

The dataset can be download by `soundata` installment.
https://github.com/soundata/soundata#quick-example

```python
pip install soundata
```

```python
import soundata

dataset = soundata.initialize('urbansound8k')
dataset.download()  # download the dataset
dataset.validate()  # validate that all the expected files are there

example_clip = dataset.choice_clip()  # choose a random example clip
print(example_clip)  # see the available data

```

The more details of the `soundata` manual book can be found here:
https://soundata.readthedocs.io/en/latest/
