# Brain_tumor_detection

Here in brain tumor classification we are taking Brain MRI images and detecting whether it has brain tumor or not.


# Data_spilting 
 We are splitting the data into train, test and validate our model.

#DATA_PREPROCESSING:
-Data preprocessing involves the transformation of the raw dataset into an understandable format.
-Preprocessing data is a fundamental stage in data miningto improve data efficiency. The data preprocessing methods directly affect the outcomes of any analytic algorithm.
-The dataset we have used are obtained from The Cancer Imaging Archive (TCIA). 
-Data set correspond to Brain MRI images together with manual FLAIR abnormality segmentation masks.
-First step is usually importing the libraries that are useful reading the dataset and performing some operations on the data.
-Initially we have used glob library to read the dataset from the drive. Later we have created a data frame from that using pandas library with first column patient ID and its path.
-Using that path address we have divided the images into two types. One is normal brain MRI images of a patient and other are the segmented brain MRI images. 
-Now, we replaced path column in the data frame with these normal images path and masked images path. Later we created a new column mask which is the labelsfor this dataset. These are created by using the opencv library.
-Each and every masked image is read by the cv2 to find the region of interests, if they are found we will assign the mask with one otherwise zero.
-Next we will have to do feature Extraction.

#FEATURE_EXTRACTION:
-Feature extraction is a part of the dimensionality reduction process, in which, an initial set of the raw data is divided and reduced to more manageable groups.So when you want to process it will be easier. 
-So Feature extraction helps to get the best feature from big data sets by select and combine variables into features, thus, effectively reducing the amount of data.
-These features are easy to process, but still able to describe the actual data set with the accuracy and originality.
-Before performing feature extraction images are split into three sets as training,validation and testing. 
-After that we are rescaling these images by using the ImageDataGenerator. 
-Till now we have made all the images in the same range 0 to 1, from now we have extract the features from the images. 
-We have used ResNet50 to extract the features from the Images.
-ResNet-50 is Residual network with 50 layers. And it is called Residual because we have Skip connections. 


#Convolution layer:
-Firstly this layer receives the input image. To extract features from that image we are going to apply a feature detector which is actually a matrix. 
-When we applied this feature detector, we are going to get a feature map which is the reduced form of the above input image. This feature map is also a matrix which consists of important features of the image.

#ReLU function:
-ReLU stands for rectified linear unit, and is a type of activation function.
-Mathematically, it is defined as y = max (0, x).
-ReLU is the most commonly used activation function in neural networks, especially in CNNs. 
-It is mainly used to maintain the nonlinearity as the features what we are getting are going in a linear, images are actually non-linear so that is the reason we have used this activation function.


#Pooling:
-A pooling layer is a new layer added after the convolutional layer. 
-Specifically, after a non-linearity (e.g. ReLU) has been applied to the feature maps output by a convolutional layer. 
-The pooling layer operates upon each feature map separately to create a new set of the same number of pooled feature maps.
-Pooling involves selecting a pooling operation, much like a filter to be applied to feature maps. The size of the pooling operation or filter is smaller than the size of the feature map.
-Pooling layer will always reduce the size of each feature map and concentrate only on the important features.

#Flattening:
-Flattening just converts this pooled feature maps to the single dimensional array.

#Fully Connected Layers:
-Fully Connected Layer is simply, feed forward neural networks.
-Fully Connected Layers form the last few layers in the network. The input to the fully connected layer is the output from the final Pooling or Convolutional Layer,zwhich is flattened and then fed into the fully connected layer.
-In general in neural networks as the number of layers increases it becomes difficult to train the network.
-So here we used callback is an object that can perform actions at various stages of training i.e., at the start or end of an epoch, before or after a single batch. 
-ReduceLROnPlateau, EarlyStopping, ModelCheckpoint and LearningRateScheduler are the classes used for callback from the obtained features we detect the tumor location.

#Detection and Classification:-
-After successfully training the model, we test the model with the test dataset.
-The model detects the brain tumor location (if any) and classifies.


