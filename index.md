---
layout: default
---

## How to quantify textures?
To extract information from a stochastic field, we need some vocabulary to describe its statistical properties, i.e., the textures in the field. However, quantifying textures is in general challenging. The most widely used vocabulary -- the power spectrum -- is concise and easy to use, but has limited ability to characterize complex strucutres, as shown in the figure below. Recently, convolutional neural networks (CNN) have shown impressive ability to perform this task. However, training a CNN involves optimization of millions to billions of parameters and a large training dataset. It also face interpretability and transparency problems, which is crucial for scientific research. What makes CNN so powerful? Is it possible to combine the advantages of both analytical statistics and CNNs? 
<br>
<img src="https://github.com/SihaoCheng/scattering_transform/blob/master/figures/fig1.png?raw=true" width="500" />

## The scattering transform
The [scattering transform](https://arxiv.org/abs/1101.2286) is a statistical tool borrowing ideas from CNNs. On the one hand, it generates a compact set of powerful summary statistics with desirable mathematical properties. On the other hand, it can be seen as a toy-model for CNNs with pre-determined filters, and can be used to decipher the inner working of CNNs. To demonstrate its power, we show image synthesis result on a variety of physical fields using the scattering transform (see the figures above and below). Image synthesis is a way to visualize what a field looks like in the eye of a particular statistic. The textural similarity between the inputs and synthesis results reveals the informativeness of the scattering statistics. For more details, please see [our paper](https://pages.jh.edu/scheng40/Draft_SihaoCheng).
<br>
<img src="https://github.com/SihaoCheng/scattering_transform/blob/master/figures/fig2.png?raw=true" width="600" />

An illustration of the definition of the scattering transform is shown in the figure below. It transforms the input field by wavelet convolutions and pointwise non-linearity (complex modulus) and repeats this combination of operations to go higher orders. Then, to obtain translation-invariant statistics, it takes the global average of the respective transformed fields. They are called `scattering coefficients'. Its structural similarity to CNNs can be clearly seen in the figure below.
<br>
<img src="https://github.com/SihaoCheng/scattering_transform/blob/master/figures/fig4.png?raw=true" width="600" />

## Interpretation
The scattering transform has very interesting visual interpretation. The first-order scattering coefficients are similar to binned power spectrum. Indeed, if we raise the modulus operation into squared modulus, these coefficients becomes exactly the binned power spectrum.
<br>
<img src="https://github.com/SihaoCheng/scattering_transform/blob/master/figures/fig5.png?raw=true" width="600" />

The second-order scattering coefficients are more interesting, as they contain substantial non-Gaussian information and can also be interpreted. After a further reduction, these coefficients can represent the sparsity and shape of features in the field:
<br>
<img src="https://github.com/SihaoCheng/scattering_transform/blob/master/figures/fig7.png?raw=true" width="400" />
<br>
<img src="https://github.com/SihaoCheng/scattering_transform/blob/master/figures/fig8.png?raw=true" width="300" />


<h1 id="Contacts">Contacts</h1>
s.cheng@jhu.edu
<br>
+1 443 207 1532
<br>
Bloomberg 506
<br>
3400 N. Charles St., Johns Hopkins University
<br>
Baltimore, MD21218, USA
