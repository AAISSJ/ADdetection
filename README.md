# ICASSP 2023 SPGC Challenge: Multilingual Alzheimer's Dementia Recognition through Spontaneous Speech 🧑‍👨‍🦳


## Description ✍️

The ADReSS-M Signal Processing Grand Challenge targets a difficult automatic prediction problem of societal and medical relevance, namely, the detection of Alzheimer's Dementia (AD). Dementia is a category of neurodegenerative diseases that entails a long-term and usually gradual decrease of cognitive functioning. While there has been much interest in automated methods for cognitive impairment detection by the signal processing and machine learning communities (de La Fuente, Ritchie and Luz, 2020), most of the proposed approaches have not investigated which speech features can be generalised and transferred across languages for AD prediction, and to the best of our knowledge no work has investigated acoustic features of the speech signal in multilingual AD detection. The ADReSS-M Challenge targets this issue by defining a prediction task whereby participants train their models based on English speech data and assess their models' performance on spoken Greek data. It is expected that the models submitted to the challenge will focus on acoustic features of the speech signal and discover features whose predictive power is preserved across languages, but other approaches can be considered.

<br>

## Our Experiments 🛠

- Unimodal 
  - Text
    - XLM : [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291)
    - mBERT : [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
  - Audio 
    - wav2vec 2.0 : [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)
    - XLS-R : [XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale](https://arxiv.org/abs/2111.09296)
- Multimodal 
  - concatenate
  - addition 
  - self-attention
  - cross-attention 


<br>

## Results 🚀

![image](https://user-images.githubusercontent.com/76966915/218398911-c1fba553-701b-448f-9e46-f42ad24ba400.png)

