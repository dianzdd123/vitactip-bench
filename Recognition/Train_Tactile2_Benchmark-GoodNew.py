# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 18:38:10 2024

@author: hp
"""
import csv
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import numpy as np
import time

class A2TrainKeras:

    def __init__(self, train_dir_path, val_dir_path, test_dir_path, batch_size, backbone, structure, dataset_ratio, logfile):
        self.logfile = logfile
        self.train_dir_path = train_dir_path
        self.val_dir_path = val_dir_path
        self.test_dir_path = test_dir_path
        self.backbone = backbone
        self.structure = structure
        self.dataset_ratio = dataset_ratio
        self.batch_size = batch_size
        self.model = self.build_model()
        self.model.compile(optimizer=optimizers.Adam(lr=0.0005),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.train_gen, self.val_gen, self.test_gen = self.get_data_generators()

    def write_to_file(self, file_path, text):
        with open(file_path, 'a') as file:
            file.write(text + '\n')

    def build_model(self):
        input_shape = (256, 256, 3)  # Adjust input shape to your image size
        inputs = layers.Input(shape=input_shape)
        x = self.select_backbone(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(250, activation='softmax')(x)  # Adjust the number of classes as needed
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def build_alexnet(self, input_shape, num_classes):
        model = models.Sequential()
    
        # Layer 1: Convolutional + BatchNormalization + MaxPooling
        model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation="relu", input_shape=input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
        # Layer 2: Convolutional + BatchNormalization + MaxPooling
        model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
        # Layer 3: Convolutional + BatchNormalization
        model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
        model.add(layers.BatchNormalization())
    
        # Layer 4: Convolutional + BatchNormalization
        model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
        model.add(layers.BatchNormalization())
    
        # Layer 5: Convolutional + BatchNormalization + MaxPooling
        model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    

        return model


   
    
    def build_resnet_block(self, input_tensor, filters, kernel_size=3, stride=1, conv_shortcut=False):
        """A basic ResNet block with identity or convolutional shortcut."""
        # Main path
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
    
        # Shortcut path
        if conv_shortcut:
            shortcut = layers.Conv2D(filters, 1, strides=stride)(input_tensor)
            shortcut = layers.BatchNormalization()(shortcut)
        else:
            shortcut = input_tensor
    
        # Adding the shortcut to the main path
        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)
        return x

    def build_resnet18(self, input_shape):
        """Builds a ResNet18 architecture that mimics the AlexNet structure."""
        inputs = layers.Input(shape=input_shape)
    
        # Initial convolution similar to AlexNet
        x = layers.Conv2D(96, (11, 11), strides=4, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2)(x)
    
        # Residual blocks mimic AlexNet's structure
        x = self.build_resnet_block(x, 256, stride=1, conv_shortcut=True)  # Residual block to mimic AlexNet Layer 2
        x = layers.MaxPooling2D((3, 3), strides=2)(x)
    
        x = self.build_resnet_block(x, 384, stride=1, conv_shortcut=True)  # Residual block to mimic AlexNet Layer 3
        x = self.build_resnet_block(x, 384, stride=1, conv_shortcut=False)  # Residual block to mimic AlexNet Layer 4
    
        x = self.build_resnet_block(x, 256, stride=1, conv_shortcut=True)  # Residual block to mimic AlexNet Layer 5
        x = layers.MaxPooling2D((3, 3), strides=2)(x)
    

        
        return models.Model(inputs, x)
    
    def select_backbone(self, inputs):
        if self.backbone == 'CNN3':
            x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(64, (3, 3), activation='relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(128, (3, 3), activation='relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.GlobalAveragePooling2D()(x)  # This is valid for CNN backbones
        elif self.backbone == 'ResNet18':
            resnet18_model = self.build_resnet18(input_shape=(256, 256, 3))
            x = resnet18_model(inputs)
        elif self.backbone == 'AlexNet':
            x = self.build_alexnet((256, 256, 3), 250)(inputs)
            return x  # Return directly from AlexNet (no additional pooling)
        elif self.backbone.startswith('VGG'):
            base_model = tf.keras.applications.VGG16(include_top=False, input_tensor=inputs, weights='imagenet') if self.backbone == 'VGG16' else tf.keras.applications.VGG19(include_top=False, input_tensor=inputs, weights='imagenet')
            x = base_model(inputs, training=False)
            x = layers.GlobalAveragePooling2D()(x)  # This is valid for VGG backbones
        elif self.backbone.startswith('ResNet'):
            base_model = tf.keras.applications.ResNet50(include_top=False, input_tensor=inputs, weights='imagenet') if self.backbone == 'ResNet50' else None
            x = base_model(inputs, training=False)
            x = layers.GlobalAveragePooling2D()(x)  # This is valid for ResNet backbones
        elif self.backbone == 'DenseNet121':
            base_model = tf.keras.applications.DenseNet121(include_top=False, input_tensor=inputs, weights='imagenet')
            x = base_model(inputs, training=False)
            x = layers.GlobalAveragePooling2D()(x)  # This is valid for DenseNet backbones
        else:
            raise ValueError(f"Unknown backbone {self.backbone}")
        return x


    def get_data_generators(self):
        datagen = ImageDataGenerator(rescale=1./255)
        
        # Load the training dataset
        train_gen = datagen.flow_from_directory(
            self.train_dir_path,
            target_size=(256, 256),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True  # Ensure the data is shuffled
        )
        
        # Adjust the training generator based on the dataset ratio
        if self.dataset_ratio < 1.0:
            total_train_samples = int(train_gen.samples * self.dataset_ratio)
            selected_indices = np.random.choice(train_gen.samples, total_train_samples, replace=False)
            train_gen.filepaths = np.array(train_gen.filepaths)[selected_indices].tolist()
            train_gen.samples = total_train_samples
        
        # Load the validation dataset
        val_gen = datagen.flow_from_directory(
            self.val_dir_path,
            target_size=(256, 256),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False  # No need to shuffle validation data
        )
    
        # Load the test dataset
        test_gen = datagen.flow_from_directory(
            self.test_dir_path,
            target_size=(256, 256),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False  # No need to shuffle test data
        )
        
        return train_gen, val_gen, test_gen

    def start_training(self):
        csv_file = f'{self.backbone}-{self.structure}-{self.dataset_ratio}-training.csv'
        checkpoint = ModelCheckpoint(filepath=f'{self.backbone}-{self.structure}-{self.dataset_ratio}-best_model.h5',
                                     save_best_only=True, monitor='val_loss', mode='min')
        csv_logger = CSVLogger(csv_file, append=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

        start_time = time.time()
        history = self.model.fit(self.train_gen,
                                 epochs=EPOCHS,
                                 validation_data=self.val_gen,
                                 callbacks=[checkpoint, early_stopping, csv_logger])
        end_time = time.time()
        self.training_time = end_time - start_time
        self.actual_epochs = len(history.history['loss'])

        self.evaluate_and_log()

        # Load and process the training log using Python's csv module
        epochs, losses, val_losses = [], [], []
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                epochs.append(int(row['epoch']))
                losses.append(float(row['loss']))
                val_losses.append(float(row['val_loss']))

        # Plot training and validation loss using plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=losses, mode='lines+markers', name='Training Loss'))
        fig.add_trace(go.Scatter(x=epochs, y=val_losses, mode='lines+markers', name='Validation Loss'))
        fig.update_layout(title='Training and Validation Loss Over Epochs',
                          xaxis_title='Epochs',
                          yaxis_title='Loss',
                          legend_title='Legend')
        fig.show()

    def evaluate_and_log(self):
        start_time = time.time()
        test_loss, test_accuracy = self.model.evaluate(self.test_gen)
        end_time = time.time()
        evaluation_time = end_time - start_time

        # Calculate Frames Per Second (FPS)
        fps = self.test_gen.samples / evaluation_time

        # Get total number of parameters
        total_params = self.model.count_params()

        log_text = (
            f"Backbone: {self.backbone}, Ratio: {self.dataset_ratio}, Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, "
            f"FPS: {fps:.2f}, Total Params: {total_params}, Training Time: {self.training_time:.2f} seconds, "
            f"Epochs: {self.actual_epochs}"
        )
        self.write_to_file(self.logfile, log_text)


if __name__ == "__main__":
    EPOCHS = 200
    BS = 32
    backbones = [ 'ResNet18','AlexNet', 'CNN3', 'VGG16', 'ResNet50', 'DenseNet121']
    train_dir_path = './train_data/'
    val_dir_path = './val_data/'
    test_dir_path = './test_data/'
    
    data_ratios = [1, 0.8, 0.6, 0.4, 0.2 ]
    
    for data_ratio in data_ratios:
        for backbone in backbones:
            logfile = f'{data_ratio}_{backbone}_model_performance.txt'
            
            tf.keras.backend.clear_session()
            instance = A2TrainKeras(
                train_dir_path=train_dir_path,
                val_dir_path=val_dir_path,
                test_dir_path=test_dir_path,
                batch_size=BS,
                backbone=backbone,
                structure='flat',
                dataset_ratio=data_ratio,
                logfile=logfile
            )
            
            print(f"Training with data_ratio: {data_ratio}, backbone: {backbone}")
            instance.start_training()

