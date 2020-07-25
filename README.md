# Image Orientation Classification
- Image orientation classification using given vectorized 192 pixels of the image.

### Using Neural Network
- Normalized data by dividing each pixel value with 255 so that all feature values are between 0 and 1
- Instead of using for loops to calculate values at each neuron, doing matrix multiplication for quick calculations.
- Therefore weights are saved in a 2d numpy array
- Each weight matrix has a dimension of (#neurons in previous layer X #neurons in next layer)
- Intialized weights by random values following normal distribution and dividing them by square root of 1 over #neurons in input layer which is a Xaviers initializer

- Model Architecture used:
<img src="images_for_readme/nnet_arch.PNG" />
Epochs = 20, Learning rate = 0.005

- Using above model architecture, did stochastic gradient descent to update the weights
- Best model is picked based on high validation accuracy after all epochs
- Final model accuracy is about 78% on train data, validation and about 76% on test data
- Time taken by the best model to run is little over 3 minutes
- Various experiments performed:
<img src="images_for_readme/nnet_exp.PNG" />

### KNN
- KNN has two parameters:
  - K – Number of nearest neighbors.
  - Distance function ( Euclidean and Manhattan )
- We have experimented with K values from 1 to 15 for both distance functions. And 
- Got better results with Euclidean Distance over Manhattan (as Euclidian is true distance between points.)
- Highest accuracy when K = 11 (But accuracy did not vary much for K values greater than 4. 
- Runtime for all the K-values is almost similar because major part of time is taken for calculating distances.

<img src="images_for_readme/KNN_Accuracy.PNG" />
<img src="images_for_readme/KNN_Runtime.PNG" />

### Decision Tree:
- Algorithm:
  - Used ID3 algorithm with Entropy function and Information gain as metrics.
  - Find the best split value for all the attributes i.e. with max information gain. To reduce run time we have considered 5 random cutoffs between 0 to 255.
  - Based on that, find the attribute that has the maximum information gain by finding the change in entropy due to split. And save it in the Tree
  - Now as per the split, separate the dataset into two parts - Left and Right and then recursively find the attribute with maximum gain with repeating the steps above.
  - At the end of the recursion we will have the model built with leaf nodes representing the class variables - ‘0’, ‘90’, ‘180’, ‘270’
  - After the tree is built, we save the tree in “tree_model.txt” file using pickle package.
- Parameters
  - K – Maximum depth of the tree.
  - Number of thresholds values to check for each attribute.
 
<img src="images_for_readme/DecisionAcc.PNG" />
<img src="images_for_readme/DecisionRun.PNG" />


- Error analysis: Image classification is failing for all the 3 classifiers when (Below are few misclassified examples)
  - Images are mostly natural
  - Images are taken from the ariel view
  - Images are symmetric
  - Interestingly 3rd image of 180 degree rotated is already a mirror image because of water (Fig i)

| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604" alt="Image 1" src="images_for_readme/n3.PNG"> Fig a: Error when rotated 0 degrees |  <img width="1604" alt="Image 2" src="images_for_readme/a2.PNG">Fig b: Error when rotated 0 degrees|<img width="1604" alt="Image 3" src="images_for_readme/a1.PNG">Fig c: Error when rotated 0 degrees|
|<img width="1604" alt="Image 4" src="images_for_readme/n901.PNG">  Fig d: Error when rotated 90 degrees|  <img width="1604" alt="Image 5" src="images_for_readme/n902.PNG">Fig e: Error when rotated 90 degrees|<img width="1604" alt="Image 6" src="images_for_readme/n903.PNG">Fig f: Error when rotated 90 degrees|
|<img width="1604" alt="Image 4" src="images_for_readme/n1801.PNG">  Fig g: Error when rotated 180 degrees|  <img width="1604" alt="Image 5" src="images_for_readme/n1802.PNG">Fig h: Error when rotated 180 degrees|<img width="1604" alt="Image 6" src="images_for_readme/n1803.PNG">Fig i: Error when rotated 180 degrees|
|<img width="1604" alt="Image 4" src="images_for_readme/n2701.PNG">  Fig j: Error when rotated 270 degrees|  <img width="1604" alt="Image 5" src="images_for_readme/n2702.PNG">Fig k: Error when rotated 270 degrees|<img width="1604" alt="Image 6" src="images_for_readme/n2703.PNG">Fig l: Error when rotated 270 degrees|

