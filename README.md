# AURAL
AURAL: (Advanced Understanding and Recognition of Audio Logic)
![Audio-Waveform](./images/audio_waveform.png)

<!-------------------------------------------------------------------------------------------->
<div align="center">

## 🎧 From Noise to Harmony: An AI Journey in Audio Enhancement 🚀

</div>

---
<!-------------------------------------------------------------------------------------------->
<div align="center">

### Abstract 📜

</div>

In the rapidly evolving landscape of digital audio, one hurdle consistently surfaces—**the enhancement of audio recordings captured with low-quality devices**, such as phone microphones. Our research pivots around this challenge, proposing a state-of-the-art deep learning model that elevates the audio quality, transforming phone-quality audio to studio-quality sound 🎤. Trained on a large dataset of parallel recordings—simultaneously captured on both professional studio microphones and low-quality phone microphones—our model signifies a remarkable stride in the pursuit of high-quality sound recordings, accessible to all.

> **Keywords**: Audio Quality Enhancement, Deep Learning, Audio Recordings, U-Net Architecture, Generative Adversarial Network

</div>

## Table of Contents
---
1. [Related Work](#related-work)
2. [Methodology](#methodology)
3. [Experiments and Results](#experiments-and-results)
4. [Discussion](#discussion)
5. [Conclusion and Future Work](#conclusion-and-future-work)
6. [References](#references)

<div align="justify">

---
<!-------------------------------------------------------------------------------------------->

<div align="center">

### Introduction 🚪

</div>

Acquiring high-quality audio recordings is conventionally a daunting task, necessitating professional equipment and acoustically treated environments. This creates a significant barrier for aspiring musicians and podcasters. To shatter this barrier, we introduce a method to upgrade low-quality audio to studio-like quality using deep learning. Our technique bridges the gap between phone-recorded audio and studio-recorded sound, making high-quality audio more accessible.

---
<!-------------------------------------------------------------------------------------------->

<div align="center">

### Related Work 📚

</div>

The arena of **audio quality enhancement** is a vast and mature field of research. Traditional audio processing techniques have focused on noise reduction strategies, such as spectral subtraction and Wiener filtering [[Boll, S. F. (1979), Lim, J. S., & Oppenheim, A. V. (1979)]](https://ieeexplore.ieee.org/abstract/document/1456459). The incorporation of psychoacoustic models helped prioritize the preservation of certain sounds during the denoising process, reflecting the nuanced human perception of sound [[Zwicker, E., & Fastl, H. (2013)]](https://link.springer.com/book/10.1007/978-3-662-05008-1).

However, the dawn of deep learning kindled a fresh direction in audio processing. Researchers have experimented with CNNs, RNNs, and more recently, Transformer-based models for a variety of audio tasks, including sound classification, source separation, and denoising [[Pascual, S., Bonafonte, A., & Serrà, J. (2017)]](https://arxiv.org/abs/1703.09452). Notably, U-Net architectures have shown promise in audio source separation tasks, excelling in the preservation of detailed features [[Jansson, A., Humphrey, E., Montecchio, N., Bittner, R., Kumar, A., & Weyde, T. (2017)]](https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf).

The application of Generative Adversarial Networks (GANs) in the audio domain, specifically WaveGAN, is a testament to the potential of GANs in generating raw audio waveforms [[Donahue, C., McAuley, J., & Puckette, M. (2019)]](https://openaccess.thecvf.com/content_CVPR_2019/html/Donahue_Adversarial_Audio_Synthesis_CVPR_2019_paper.html). Similarly, deep learning models, like Transformer-based models, have been effectively employed in automatic music generation and instrument recognition tasks [[Huang, C. Z. A., Vaswani, A., Uszkoreit, J., Simon, I., Hawthorne, C., Shazeer, N., ... & Chen, D. (2018)]](https://arxiv.org/abs/1809.04281).

Our research seeks to enhance these deep learning techniques further by offering an innovative approach to transform low-quality audio into high-quality audio. Our methodology uniquely blends a U-Net based autoencoder with a GAN framework, ensuring the generated high-quality audio is convincingly realistic. This novel combination of techniques represents an unexplored avenue in the realm of audio quality enhancement.

---
<!-------------------------------------------------------------------------------------------->

<div align="center">

### Methodology 🛠️

</div>

Our approach to upgrading low-quality audio to high-quality hinges on several key strategies:

- We use a dual-stage autoencoder with a **U-Net architecture** trained on extensive paired low-quality and high-quality audio recordings.
- The first stage targets the removal of noise and distortion from the low-quality input, acting as a **denoising autoencoder**.
- The second stage is a **generative autoencoder** designed to imbue the denoised audio with the rich characteristics of high-quality, studio-recorded audio.
- The **Generative Adversarial Network (GAN) framework** is adopted to ensure the high-quality audio generated by our model is convincingly authentic.
- Our model also considers additional features from the audio, such as pitch and rhythm, helping capture more nuancedThe user asked: I'm trying to integrate an audio processing feature that automatically applies an EQ (Equalizer) effect to a track based on a reference track's EQ spectrum. I want the AURAL AI to understand the EQ spectrum of the reference track, adjust the EQ of the current track to match the reference. The user can specify whether they want AURAL to analyze the whole track or just a specific segment. The implementation should be in Python. Could you guide me on how to implement this?
