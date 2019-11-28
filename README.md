# GAN-TensorFlow 2.0
Tensorflow2.0 implementation of GAN.

<center><img src="./images/basic_gan.gif" width="320" height="240"></center>

---  

## Requirements
- tensorflow 2.0
- python 3
- numpy
- For make GIF & plot
  - glob
  - imageio
  - matplotlib

---

## GAN Model
### **Generator**
<center><img src="./images/generator.png" width="50%" height="50%"></center>

### **Discriminator**
<center><img src="./images/discriminator.png" width="50%" height="50%"></center>

---

## Documentation
### Download Dataset
MNIST dataset will be downloaded automatically.
```
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
```

### Training GAN
Use `basic_gan.py` to train a basic GAN network.

---

## Results
### epoch = 1
<center><img src="./images/image_at_epoch_0001.png"></center>

### epoch = 50
<center><img src="./images/image_at_epoch_0050.png"></center>

### epoch = 100
<center><img src="./images/image_at_epoch_0100.png"></center>

### epoch = 200
<center><img src="./images/image_at_epoch_0200.png"></center>

---

## Reference
 [Ian J. Goodfellow. *Generative Adversarial Nets*, NIPS2014.](https://arxiv.org/abs/1406.2661).
