# DeepWeeds_classification_tensorflow
Benchmarks tensorflow.keras models on the DeepWeeds dataset. This code is modified from https://github.com/AlexOlsen/DeepWeeds.
The download location of the dataset download is also found at https://github.com/AlexOlsen/DeepWeeds.

Ensure that the Root Directory has the following file structure:

./models
./images
./data_split
./labels
./outputs
deepweeds3.py
download_models.py
split_dataset.py

Ensure all of the images are placed in ./images. Once split_dataset.py is run. The train, validation and test splits are seen in ./data_split. Once download_models.py has been run, the ./models directory will be populated with tensorflow.keras models in the .hdf5 format. Other arbitrary deep learning encoders which accept (224x244) input size with .hdf5 format can also be trained. Once the previous steps are complete, deepweeds3.py can be run with commands such as --compute_overall_summary where 1 creates a .csv file which summarizes the testing performance of previously benchmarked models.


