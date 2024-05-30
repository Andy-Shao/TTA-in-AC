# Document info
https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5

# dataset link
https://urbansounddataset.weebly.com/urbansound8k.html

The dataset can be downloaded by `sound at` instalment.
https://github.com/soundata/soundata#quick-example

```python
pip install sound data
```

```python
import sound data

dataset = soundata.initialize('urbansound8k')
dataset.download()  # download the dataset
dataset.validate()  # validate that all the expected files are there

example_clip = dataset.choice_clip()  # choose a random example clip
print(example_clip)  # see the available data

```

More details of the `Soundata` manual book can be found here:
https://soundata.readthedocs.io/en/latest/

The data source download link is:
!wget  https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz
