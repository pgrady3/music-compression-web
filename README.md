# Introduction (Sherry)

* What is music compression? 
* Traditional methods of music compression?
* What are some common bitrates, find a few audio samples at different bitrates. What are the traditional methods for music compression
* Frequency and time domain analysis of a song (short intro to STFT)

# Motivation

Traditional methods of music compression have a deterministic algorithm, which relies on identifying features and patterns (in the frequency domain). As deep networks are really good at capturing complex patterns, we are trying to analyse the ability to condense a song's patterns into a smaller dimension space.

Once we have the latent space, we want to build models on the latent space as opposed to the original input space to build classification models.

# Dataset

We use the [FMA dataset](https://github.com/mdeff/fma). For this project we use *small* version of the dataset containing 8000 songs from 8 genre categories. We used a 70-30 split between train and test set.


# Unsupervised Audio Compression

A deep autoencoder is a special type of feedforward neural network which can be used in denoising and compression [2]. In this architecture, the network consists of an encoder and decoder module. The encoder learns to compress a high-dimensional input X to a low-dimensional latent space z. This "bottleneck" forces the information in X to be compressed. The decoder then attempts to faithfully reconstruct the output with minimal error. Both the encoder and decoder are implemented as convolutional neural networks.

Clearly, it is impossible to reconstruct the input with zero error, so the network learns a lossy compression. The network can discover patterns in the input to reduce the data dimensionality required to fit through the bottleneck. The network is penalized with an L2 reconstruction loss. This is a completely unsupervised method of training that provides very rich supervision.

![Autoencoder](ae.png)

## Frequency domain

### Model details 
TODO: Add Diagram

### Loss function 

Music is fundamentally subjective. Thus generating a quantitatively evaluation metric for our compression algorithm is very difficult. It is not possible to naively compare the reconstructed time domain signals, as completely different signals can sound the same. For example, phase shift, or small uniform frequency shifts are imperceptible to the human ear. A naive loss in the time domain would heavily penalise this.

![Phase Shift](phase_shift.png)

On the other hand, a time domain loss does not adequately capture high frequencies and low volumes. As human perception of sound is logarithmic, and low frequencies typically have higher amplitude, a time domain loss under-weights high frequencies and results in a muffled, underwater-sounding output.

We follow the approach of [1] and instead use an RMSE metric by directly comparing the frequency spectra across time. This has the benefit of considering low amplitudes and high frequencies, and is perceptually much closer.

**Original Spectrogram**

![Original Spectrogram](original_spect.png)

**Reconstructed Spectrogram**

![Reconstructed Spectrogram](reconst_spect.png)

We then use a simple RMSE metric to compare the reference and reconstruction

![RMSE Loss](rmse_loss.png)


## Time domain

Our main motivation for this approach is to build an end-to-end network so that it can potentially learn a more compressed representation. This approach is inspired from computer vision where people moved from a classical pipeline of feature design to end-to-end deep model.

### Model details
![time_domain_autoencoder](model_diagrams/time_autoencoder.jpeg)

### Loss functions
Even though an RMSE loss in the time domain is not the best choice from a point of view of audio perception, we found that it worked better than RMSE loss in spectral or log-spectral domain

# Music Genre Classification

We took the latent space obtained from time-domain compression model and added more CNNs and FC layers on top of it to perform genre classification on 8 classes.

### Model details

# Results

## Time-domain autoencoder for compression

**Latent space**
Samples from the test set:

| ![latent_time_sample_1](results/time_autoencoder/latent_space/samples/3_latent.png) | ![latent_time_sample_2](results/time_autoencoder/latent_space/samples/19_latent.png)  |
| -- | -- |
| ![latent_time_sample_3](results/time_autoencoder/latent_space/samples/31_latent.png) | ![latent_time_sample_4](results/time_autoencoder/latent_space/samples/49_latent.png)  |

Overall space for the test set:
![time_overall_latent](results/time_autoencoder/latent_space/overall.png)

## Audio samples

**Sample 1**

![Sample Audio](results/time_autoencoder/compression/sample0/plots.png)



**Sample 2**

![Sample Audio](results/time_autoencoder/compression/sample1/plots.png)




## Genre classification

* Overall accuracy on test set: 45%

  * Using majority voting on 5 2-second segments of the song
  * 42% percent using just a single 2 second segment

  

   ![confusion_matrix](results/classification/precision_confusion.png)


- Audio samples from compression
- Compression results
- Loss curves during training


# Discussiona and Conclusions

# Citations

- [1] Roche, Fanny, et al. "Autoencoders for music sound modeling: a comparison of linear, shallow, deep, recurrent and variational models." arXiv preprint arXiv:1806.04096 (2018).

- [2] Vincent, Pascal, et al. "Extracting and composing robust features with denoising autoencoders." Proceedings of the 25th international conference on Machine learning. ACM, 2008.