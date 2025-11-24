#SKIN DISEASE CLASSIFIER

The objective of this project is to develop and implement a robust deep learning solution for the automated classification of common dermatoscopic images. By leveraging a
pre-trained convolutional neural network (CNN) architecture and fine-tuning it for seven
distinct skin lesion categories, the system provides rapid and quantitative diagnostic predictions based on image analysis
-DATA SET
-INSTALLATION GUIDE
-MODEL ARCHITECTURE
##DATA SET
* [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
* [ISIC Archive](https://gallery.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery?filter=%5B%5D)
  
##INSTALLATION

* Clone this repository
* Download a test image and save it in the same folder as the model and predict file
* open the folder in a IDE and open predict_new.py and run it
* When prompted add the address of your test image
  
##Model Architecture

The solution utilizes the principle of Transfer Learning to expedite training and improve generalization, given the specialized nature of dermatoscopic images.

###Backbone Architecture: ResNet-50

The core of the model is a pre-trained ResNet-50 (Residual Network with 50 layers). ResNet is chosen for its superior performance in complex image recognition tasks, primarily due to its use of residual blocks.  These blocks employ skip connections that allow gradients to flow easily through the network, mitigating the vanishing gradient problem in very deep architectures.

###Architecture Modification (Fine-Tuning)
The ResNet-50 model, originally trained on the 1000 classes of the ImageNet dataset, is fine-tuned for the seven lesion classes using the combined HAM10000 and ISIC Archive datasets:

* Frozen Layers (Feature Extraction): The initial convolutional layers (up to the penultimate residual block) are frozen. These layers are responsible for extracting fundamental features like edges, textures, and shapes.

* Global Pooling: A Global Average Pooling (GAP) layer replaces the original model's classification head, condensing the complex feature maps into a fixed-size feature vector.

* Classification Head: A new, shallow classification head is added. This typically consists of one or two Dense (Fully Connected) layers, followed by the final output layer:

    * Output Layer: A Dense layer with 7 units (equal to the number of target classes).

    * Activation: The softmax activation function is used to produce a probability distribution across the seven skin lesion categories.

