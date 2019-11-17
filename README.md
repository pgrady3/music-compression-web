# Deep Autoencoders for Music Compression and Classification

# Introduction (Sherry)

What is music compression? What are some common bitrates, find a few audio samples at different bitrates. What are the traditional methods for music compression

What is a STFT, what is a song spectrogram

# Dataset

# Approach

## Unsupervised Audio Compression









## Music Genre Classification


Frequency autoencoder. Learn image to image translation

Signal autoencoder, attempt to learn fourier 

Supervised training for classification

## Compression Evaluation Metric

Music is fundamentally subjective. Thus generating a quantitatively evaluation metric for our compression algorithm is very difficult. It is not possible to naively compare the reconstruced time domain signals, as completely different signals can sound the same. For example, phase shift, or small uniform frequency shifts are imperceptible to the human ear. A naive loss in the time domain would heavily penalize this.

![Phase Shift](phase_shift.png)

On the other hand, a time domain loss does not adequately capture high frequencies and low volumes. As human perception of sound is logarithmic, and low frequencies typically have higher amplitude, a time domain loss under-weights high frequencies and results in a muffled, underwater-sounding output.

We follow the approach of [1] and instead use an RMSE metric by directly comparing the frequency spectra across time. This has the benefit of considering low amplitudes and high frequencies, and is perceptually much closer.

Original Spectrogram

![Original Spectrogram](original_spect.png)

Reconstructed Spectrogram

![Reconstructed Spectrogram](reconst_spect.png)

We then use a simple RMSE metric to compare the reference and reconstruction

![RMSE Loss](rmse_loss.png)

# Results

- Audio samples from compression
- Compression results
- Latent space visualization
- Latent space T-sne plot (would be cool!)
- Classifier confusion matrix
- Loss curves during training


# Discussiona and Conclusions

# Citations

- [1] Roche, Fanny, et al. "Autoencoders for music sound modeling: a comparison of linear, shallow, deep, recurrent and variational models." arXiv preprint arXiv:1806.04096 (2018).