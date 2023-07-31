<!-------------------------------------------------------------------------------------------->
<div align="center">

# AURAL: (Advanced Understanding and Recognition of Audio Logic)

<img src="waveform.gif" alt="Audio-Waveform" width="40%" height="40%">

</div>

<!-------------------------------------------------------------------------------------------->
<div align="center">

## 🎧 From Noise to Harmony: An AI Journey in Audio Enhancement 🚀

</div>

<!-------------------------------------------------------------------------------------------->

<div align="center">

### Abstract 📜

</div>

In the rapidly evolving landscape of digital audio, one hurdle consistently surfaces—**the enhancement of audio recordings captured with low-quality devices**, such as phone microphones. This paper addresses this challenge by presenting a novel deep learning model that is capable of transforming these low-quality audio recordings to simulate the high-quality acoustics of a professional recording studio 🎤.

Our model is trained on an extensive dataset of parallel recordings, each comprising the same performance captured with both a professional studio microphone and a low-quality phone microphone. The objective of our work is to democratize access to high-quality sound recordings, making them accessible to amateur musicians, podcasters, and everyone in between.

The results from our study indicate significant improvements in audio quality, effectively bridging the gap between professional studio recordings and recordings made using everyday devices. This stride in audio quality enhancement paves the way for future research in this area.

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

## 🧪 Methodology 

</div>

Our innovative approach to transforming low-quality audio recordings into high-quality ones involves several key strategies. We construct a dual-stage autoencoder with a U-Net architecture. This model is trained on an extensive dataset of paired low-quality and high-quality audio recordings. The first stage of the autoencoder acts as a denoising autoencoder, targeting the removal of noise and distortion from the low-quality input. The second stage is a generative autoencoder designed to imbue the denoised audio with the characteristics of high-quality studio-recorded audio. 

The input to the model is the mel-spectrogram of the low-quality audio, and the target output is the mel-spectrogram of the high-quality audio. The model is trained to minimize the difference between its output and the actual high-quality audio. In addition to the U-Net autoencoder, we adopt the Generative Adversarial Network (GAN) framework to facilitate the generation of more realistic high-quality audio. The generator, which transforms low-quality audio into high-quality audio, is paired against a discriminator that differentiates between authentic high-quality audio and the generator's output. To enhance the performance of the U-Net autoencoder, we condition the model on additional features extracted from the audio, such as pitch or rhythm. This helps our model capture more nuanced characteristics of the sound source. Our model is also trained on auxiliary tasks, including source separation and pitch detection, alongside the primary task of transforming low-quality to high-quality audio. This multi-task learning approach enables our model to learn more robust representations of the audio data. 

To diversify our training data and improve the model's generalization, we employ data augmentation techniques like time stretching, pitch shifting, and adding background noise. We leverage transfer learning by initializing our model with weights from a model pre-trained on a related task. This not only accelerates the learning process but also provides our model with a foundation of useful audio features. In situations where paired high-quality and low-quality recordings are limited, we pre-train our model in a self-supervised manner using unpaired data. This involves tasks like predicting the next frame of a spectrogram or masking and then reconstructing parts of the spectrogram. We incorporate psychoacoustic features, such as loudness or sharpness, into our model. 

These features provide additional insights into how the audio is perceived by humans, thereby improving the subjective quality of our output audio. Through constant evaluation using objective metrics and subjective listening tests, we iterate and refine our approach, ensuring our model's performance is optimized. We are mindful of the limitations and assumptions inherent in our approach and take these into account during our development process.

---

<div align="center">

### 🔬 Experiments and Results

</div>

We evaluate our model on a separate test set of low-quality and high-quality recording pairs. Objective measures such as Signal-to-Noise Ratio (SNR) and Perceptual Evaluation of Speech Quality (PESQ) show significant improvement in audio quality. Subjective listening tests also indicate an enhancement in sound quality, with listeners often unable to distinguish our model's output from the actual high-quality recording.

---

<div align="center">

### 💡 Discussion

</div>

Our results demonstrate the potential of deep learning for audio quality enhancement. However, the model's performance varies depending on the type of sound source and the quality of the input audio. Further improvements might be achieved with a larger and more diverse training dataset or modifications to the model architecture.

---

<div align="center">

### 🎯 Conclusion and Future Work

</div>

This paper presents a novel approach to enhancing audio quality using deep learning. While our results are promising, there is still room for improvement. Future work could explore different model architectures, training techniques, or feature representations. Our hope is that this work will inspire further research in this area, with the goal of making high-quality sound recordings accessible to all.

---

<div align="center">

### 📌 References

</div>

1. **Audio quality enhancement, noise reduction, and traditional audio processing techniques**: 
   - Boll, S. F. (1979). Suppression of acoustic noise in speech using spectral subtraction. IEEE transactions on acoustics, speech, and signal processing, 27(2), 113-120. [Link](https://ieeexplore.ieee.org/abstract/document/1163209)
   - Lim, J. S., & Oppenheim, A. V. (1979). Enhancement and bandwidth compression of noisy speech. Proceedings of the IEEE, 67(12), 1586-1604. [Link](https://ieeexplore.ieee.org/abstract/document/1456459)
2. **Psychoacoustic models for sound preservation during denoising**: 
   - Zwicker, E., & Fastl, H. (2013). Psychoacoustics: Facts and models. Springer Science & Business Media. [Link](https://link.springer.com/book/10.1007/978-3-662-05008-1)
3. **Deep learning for audio tasks, including sound classification, source separation, and denoising**: 
   - Pascual, S., Bonafonte, A., & Serrà, J. (2017). SEGAN: Speech enhancement generative adversarial network. arXiv preprint arXiv:1703.09452. [Link](https://arxiv.org/abs/1703.09452)
4. **U-Net architectures for audio source separation**: 
   - Jansson, A., Humphrey, E., Montecchio, N., Bittner, R., Kumar, A., & Weyde, T. (2017). Singing voice separation with deep u-net convolutional networks. ISMIR, 323-332. [Link](https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf)
5. **Generative Adversarial Networks (GANs) in the audio domain, WaveGAN**: 
   - Donahue, C., McAuley, J., & Puckette, M. (2019). Adversarial audio synthesis. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 5680-5689. [Link](https://openaccess.thecvf.com/content_CVPR_2019/html/Donahue_Adversarial_Audio_Synthesis_CVPR_2019_paper.html)
6. **Deep learning in the music domain for tasks such as automatic music generation and instrument recognition**: 
   - Huang, C. Z. A., Vaswani, A., Uszkoreit, J., Simon, I., Hawthorne, C., Shazeer, N., ... & Chen, D. (2018). Music transformer: Generating music with long-term structure. arXiv preprint arXiv:1809.04281. [Link](https://arxiv.org/abs/1809.04281)
7. **Transformer-based models for generating symbolic music**: 
   - Huang, C. Z. A., Vaswani, A., Uszkoreit, J., Simon, I., Hawthorne, C., Shazeer, N., ... & Chen, D. (2018). Music transformer: Generating music with long-term structure. arXiv preprint arXiv:1809.04281. [Link](https://arxiv.org/abs/1809.04281)

```bash
AURAL_sound_regeneration/
│
├── data/
│   ├── raw/
│   │   ├── audio1.wav
│   │   ├── audio2.wav
│   │   └── ...
│   │
│   ├── preprocessed/
│   │   ├── audio1.npy
│   │   ├── audio2.npy
│   │   └── ...
│   │
│   └── uncompressed/
│       ├── audio1.wav
│       ├── audio2.wav
│       └── ...
│
├── models/
│   ├── noise_reduction/
│   │   ├── model1.h5
│   │   └── ...
│   ├── dynamic_compression/
│   │   ├── model2.h5
│   │   └── ...
│   ├── frequency_expansion/
│   │   ├── model3.h5
│   │   └── ...
│   ├── source_separation/
│   │   ├── model4.h5
│   │   └── ...
│   └── frequency_generation/
│       ├── model5.h5
│       └── ...
│
├── scripts/
│   ├── data_preprocessing/
│   │   ├── load_audio.py
│   │   ├── normalize_audio.py
│   │   ├── convert_to_mono.py
│   │   └── resample_audio.py
│   │
│   ├── model_training/
│   │   ├── train_noise_reduction_model.py
│   │   ├── train_dynamic_compression_model.py
│   │   ├── train_frequency_expansion_model.py
│   │   ├── train_source_separation_model.py
│   │   └── train_frequency_generation_model.py
│   │
│   ├── model_evaluation/
│   │   ├── evaluate_noise_reduction_model.py
│   │   ├── evaluate_dynamic_compression_model.py
│   │   ├── evaluate_frequency_expansion_model.py
│   │   ├── evaluate_source_separation_model.py
│   │   └── evaluate_frequency_generation_model.py
│   │
│   ├── audio_regeneration/
│   │   ├── apply_noise_reduction.py
│   │   ├── apply_dynamic_compression.py
│   │   ├── apply_frequency_expansion.py
│   │   ├── apply_source_separation.py
│   │   ├── apply_frequency_generation.py
│   │   └── mix_and_master.py
│   │
│   └── main.py
│
└── README.md
```

<div align="center">

## 🧑‍💼 Authorship

</div>

This work is conducted by:

- **Gabriel A. Lacroix**

  🏢 Independent Researcher, Musician and Developer
  
  📧 gabemakesrecords@example.com
  
  🌐 [Connect on LinkedIn!](https://www.linkedin.com/in/green-alderson-56b930273/)

</div>

---

