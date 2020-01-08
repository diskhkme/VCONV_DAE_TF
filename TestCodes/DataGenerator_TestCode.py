from DataGenerator import DataGenerator

training_generator = DataGenerator('../Data/ModelNet30/train_files.txt',
                                   batch_size=32, dim=(30, 30, 30),
                                   n_classes=31, shuffle=True)

training_generator.__getitem__(0)