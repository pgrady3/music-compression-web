# music-compression-web

# Introduction (Sherry)

What is music compression? What are some common bitrates, find a few audio samples at different bitrates. What are the traditional methods for music compression

What is a STFT, what is a song spectrogram

# Approach

Frequency autoencoder. Learn image to image translation

Signal autoencoder, attempt to learn fourier 

## Evaluation Metric

Music is fundamentally subjective. Thus generating a quantitatively evaluation metric for our compression algorithm is very difficult. It is not possible to naively compare the reconstruced time domain signals, as completely different signals can sound the same. For example, phase shift, or small uniform frequency shifts are imperceptible to the human ear, but a naive loss in the time domain would heavily penalize this.

![Phase Shift](phase_shift.png)




# Results

- Audio samples from compression
- Compression score sheet




# Citations

- [1] Roche, Fanny, et al. "Autoencoders for music sound modeling: a comparison of linear, shallow, deep, recurrent and variational models." arXiv preprint arXiv:1806.04096 (2018).