# Binary-Classification-using-CNN-and-Data-parallelism-with-MPI
Binary data classification using TensorFlow and Keras in python. Data parallelism using MPI for data augmentation, model training and testing.

# Steps to use the program

1. Download/ clone the repository.
2. Navigate to the respective folder in PC and create 4 seperate folders named "logs", "models", "dog_cat", "data" and "augmented". The model will be saved in the "models" folder and logs in the "logs" folder.

3. Download the data from the dataset and put them in the "dog_cat" folder by creating subfolders with their label name. For eg: in the data folder create a folder called "dog" and inside the dog folder create another folder by the same name "dog" and put all the downloaded images of dog in here. **NOTE: The data augmentation program would work properly only if this is done properly**.

4. Once after the data folder is ready with all the data files go to the augmented folder and create separate folders for separate labels, for eg: one folder named "dog", one named "cat". These label names should be same as the name in the "dog_cat" folder. **NOTE: It is the data in the augmented folder that is used as the main data for training and testing the model**.

5. Now we can successfully run the augmentation by just changing the number of iterations to get the desired number of data generated.

6. Now we can just run the data_separation.py file to separate the data to acheive data parallelism. The total data would be separated into a number of folders equal to the number of threads giving during in MPI execution command. Now each data subfolder will have separate class of data. **NOTE: Currently the classification would work for only 4 data folders. So give 4 threads. If you change the number of threads then you are expected to make neccessary changes in the pet_classfier_prl.py file.

7. Finally, run the classifier program by changing the desired number of iterations.

# Commands to run the program

1. mpiexec -np 4 python data_separation.py
2. mpiexec -np 5 python pet_classifier_prl.py
