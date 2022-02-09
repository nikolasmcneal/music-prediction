# music-prediction
Testing a deep neural network's ability to predict audio and classify musicality, and analyzing the semantic representations of neurons in the network's stacked LSTM's.
 
## Overview

In this study, we show that a predictive coding-inspired deep neural network demonstrates musicality perception similar to humans. We scale the predictive neural network [PredNet](https://arxiv.org/abs/1605.08104)[1] for audio prediction and test its predictive capabilities on a series of musical and non-musical audio clips, as defined by human subject data from [Randall and Greenberg (2016)](https://www.researchgate.net/publication/344596330_Principal_Components_Analysis_of_Musicality_in_Pitch_Sequences). We show that it performs better on musical sequences than non-musical sequences, and we observe a "context effect" where the mean squared error (MSE) decreases throughout the duration of musical sequences, suggesting that convolutional-LSTM units provide the adequate architecture for learning the underlying attributes of musicality. In addition, we perform a quantitative analysis on the representation states of the network, seeking to gain insight into the LSTM neurons' abstractions.


## Methodology

In order to test PredNet's ability on audio prediction, we convert each audio sequence to a mel spectrogram[2] and divide the mel spectrogram in equally-sized frames, so that we can utilize a sliding window technique for frame-by-frame prediction. This approach alone, however, neglects the fact that certain input frames contain more "hint" of the subsequent note than others, providing an unfair comparison between frames and sequences. To normalize this effect, we perform our experiment eight times (as there are eight pixels width overlap between frames), each time shifting the starting point of the spectrogram by one pixel. This approach permits us to fairly compare the prediction errors of a note across sequences, given a certain number of pixels' "hint."

<img src="/img/fig-2-updated.jpg" width=100%>

We train on the [Medley-solos-DB](https://zenodo.org/record/1344103#.YgNlIPXMLzc) dataset[3] and test on the ten most musical and ten most non-musical pitch sequences from Randall and Greenberg (2016)[4]. We also test on 20% of the Medley-solos-DB dataset, using a model trained on 80% of the Medley-solos-DB dataset. 

The 20 clips from Randall and Greenberg (2016) are available in [test_data](/test_data/), and the Medley-solos-DB dataset is available on the CMU MIND cluster at **/user_data/nmcneal/music-prediction/**. We divide each clip in the Medley-solos-DB dataset into 27 frames with 8 pixels width of overlap. We divide each clip in the Randall and Greenberg dataset into 49 frames with 8 pixels width of overlap. All frames are of size 128 x 48, including two columns of black pixels padded to the sides of each frame.

In addition, we train [multiple models](/models/), with separate experiments including adding another LSTM layer, removing an LSTM layer, shortening the length of the audio sequences in the training data, and changing the size of the overall training data. We find that the number of LSTM layers plays the most significant role in attenuating the model's "context effect," where the prediction error decreases throughout the duration of musical sequences. 

When comparing different musicality categories' MSE's, we first divide each sequence's MSE by the sequence's average interval size (average magnitude of pitch change between notes). We then average each newly-scored sequence's MSE's to obtain the MSE for each category. The purpose for this normalization is that we observe a strong positive correlation between the interval size and prediction error. This is expected, as the model is hesitant to generate predictions with large note jumps. Importantly, even after this normalization, we still observe our context effect for musical sequences.


<p align="center">
  <img src="/img/pos-corr.jpg" width=50%>
</p>





## Results
We observe better performance on musical sequences than non-musical sequences from the Randall and Greenberg dataset, suggesting that predictability is a correlate of musicality. In addition, we notice a "context effect," where the prediction error decreases throughout the duration of musical sequences but not non-musical sequences. In fact, throughout the duration of certain non-musical clips, we observe a significant increase in prediction error. We therefore notice an increase in the difference between non-musical and musical error. In the following diagram, the blue line represents the average MSE of musical sequences, the orange line represents the average MSE of non-musical sequences, and the green line represents the difference between non-musical and musical MSE's. 

<p align="center">
  <img src="/img/mse-difference-detailed.jpg" width=80%>
</p>




In addition, in alignment with the findings from Randall and Greenberg, we only find scores for the range of pitch, average interval size, and standard deviation of the average interval size to correlate with predictability. Scores such as partial predictability of motive and contour do not correlate with the rankings or prediction errors.

<p align="center">
  <img src="/img/attributes.jpg" width=60%>
</p>


## Current Work 
We are currently performing a quantitative analysis on the neurons in the LSTM layers to derive insight into what information is being learned and stored. On the Randall and Greenberg dataset, indexing a neuron in the bottom-most representation layer directly in the scope of where predictions are generated shows its oscillatory behavior, where each spike corresponds to a note change, and each line corresponds to a sequence (the x-axis represents the timestep of prediction, and y-axis represents the magnitude of the neuron).  

<p align="center">
  <img src="/img/plots/RG/RG_20_sequences_neuron:_100_20.jpg" width=45%>
</p>


Sampling some of the Medley-solos-DB data, where notes do not change at a constant rate (and audio consists of real instruments instead of sine wave pure-tone frequencies), the neuron behavior is more erratic.

<p align="center">
  <img src="/img/plots/MDB/MDB_50_sequences_neuron:_100_20.png" width=45%>
</p>



On the full Medley-solos-DB dataset, we observe a slightly left-skewed distribution and are analyzing the activity within the neuron's local scope in the frames on the tails of the distribution. This will provide insight on the meaning of the magnitude of the neuron.

<p align="center">
  <img src="/img/plots/MDB/MDB_hist_sequences_neuron:_100_20.png" width=45%>
</p>


## Future Directions
While we are able to show the successful application of LSTM units for audio analysis, properly representing audio in this context remains a challenge. Our model uses mel spectograms, which only maintain the time and frequency properties of audio. Future work could expand on representing the various features of audio for audio prediction tasks.

In addition, future work could test if neural networks perform musicality prediction and classification differently when trained or tested on certain styles of audio or music.

## References

[1] Lotter, William, Gabriel Kreiman, and David Cox (Mar. 2017). “Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning”. In: URL: https://arxiv. org/pdf/1605.08104.pdf.

[2] Costa, Yandre M.G., Luiz S. Oliveira, and Carlos N. Silla Jr. (Mar. 2017). “An evaluation of Convolutional Neural Net- works for music classification using spectrograms”. In: URL: https://dl.acm.org/doi/10.1016/j.asoc.2016.12.024.

[3] Lostanlen, Vincent et al. (2018). “Medley-solos-DB: a cross- collection dataset for musical instrument recognition”. In: URL: http://doi.org/10.5281/zenodo.1344103.

[4] Randall, Richard and Adam Greenberg (July 2016). “Principal Components Analysis of Musicality in Pitch Sequences”.
