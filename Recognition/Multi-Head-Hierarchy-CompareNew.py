#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 00:07:25 2024

@author: dd
"""


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, VGG16, DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import time

EPOCHS = 200
BS = 32
DEVICE = "gpu" if tf.config.list_physical_devices('GPU') else "cpu"

class CustomDirectoryIterator(tf.keras.preprocessing.image.DirectoryIterator):
    def __init__(self, directory, image_data_generator, texture_classes, stiffness_classes, **kwargs):
        self.texture_classes = texture_classes
        self.stiffness_classes = stiffness_classes
        self.num_texture_classes = len(texture_classes)
        self.num_stiffness_classes = len(stiffness_classes)

        self.texture_class_indices = {label: idx for idx, label in enumerate(texture_classes)}
        self.stiffness_class_indices = {label: idx for idx, label in enumerate(stiffness_classes)}

        super().__init__(directory, image_data_generator, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = super()._get_batches_of_transformed_samples(index_array)
        batch_size = len(batch_x)
        
        # Initialize label arrays
        texture_labels = np.zeros((batch_size, self.num_texture_classes), dtype=np.float32)
        stiffness_labels = np.zeros((batch_size, self.num_stiffness_classes), dtype=np.float32)
        
        for i, idx in enumerate(index_array):
            # Extract the directory name
            dir_name = self.filenames[idx].split(os.path.sep)[0]
            _, texture_label, stiffness_label = dir_name.split('_')
            
            # Convert texture and stiffness labels to one-hot encoding
            texture_index = self.texture_class_indices[texture_label]
            stiffness_index = self.stiffness_class_indices[stiffness_label]
            
            texture_labels[i, texture_index] = 1.0
            stiffness_labels[i, stiffness_index] = 1.0
        
        return batch_x, [texture_labels, stiffness_labels]

class CustomImageDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode=None, batch_size=32, shuffle=True, 
                            seed=None, save_to_dir=None, save_prefix='', save_format='png',
                            follow_links=False, subset=None, interpolation='nearest',
                            texture_classes=None, stiffness_classes=None):
        
        return CustomDirectoryIterator(
            directory, self, texture_classes=texture_classes, stiffness_classes=stiffness_classes,
            target_size=target_size, color_mode=color_mode, classes=classes, class_mode=class_mode,
            data_format=self.data_format, batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format, 
            follow_links=follow_links, subset=subset, interpolation=interpolation)

class ComparisonStudy:

    def __init__(self, train_dir_path, val_dir_path, test_dir_path, batch_size, logfile):
        self.logfile = logfile
        self.train_dir_path = train_dir_path
        self.val_dir_path = val_dir_path
        self.test_dir_path = test_dir_path
        self.batch_size = batch_size
        self.datagen = CustomImageDataGenerator(rescale=1./255, validation_split=0.2)
        self.model = None
        self.training_time = None
        self.actual_epochs = None
        self.backbone = None
        self.dataset_ratio = None

        # Get texture and stiffness classes from directory structure
        self.texture_classes = sorted(set([dir.split('_')[1] for dir in os.listdir(train_dir_path)]))
        self.stiffness_classes = sorted(set([dir.split('_')[2] for dir in os.listdir(train_dir_path)]))

    def get_data_generators(self, data_ratio):
        train_gen = self.datagen.flow_from_directory(
            self.train_dir_path,
            target_size=(256, 256),
            batch_size=self.batch_size,
            class_mode=None,
            shuffle=True,
            texture_classes=self.texture_classes,
            stiffness_classes=self.stiffness_classes
        )
    
        val_gen = self.datagen.flow_from_directory(
            self.val_dir_path,
            target_size=(256, 256),
            batch_size=self.batch_size,
            class_mode=None,
            shuffle=False,
            texture_classes=self.texture_classes,
            stiffness_classes=self.stiffness_classes
        )
    
        test_gen = self.datagen.flow_from_directory(
            self.test_dir_path,
            target_size=(256, 256),
            batch_size=self.batch_size,
            class_mode=None,
            shuffle=False,
            texture_classes=self.texture_classes,
            stiffness_classes=self.stiffness_classes
        )
    
        # Adjust the training generator based on the data ratio
        if data_ratio < 1.0:
            total_train_samples = int(train_gen.samples * data_ratio)
            selected_indices = np.random.choice(train_gen.samples, total_train_samples, replace=False)
            train_gen.filepaths = np.array(train_gen.filepaths)[selected_indices].tolist()
            train_gen.samples = total_train_samples
    
        return train_gen, val_gen, test_gen

    def write_to_file(self, logfile, text):
        with open(logfile, 'a') as file:
            file.write(text + '\n')

    
    
    def build_cnn_base(self, input_shape, layers_count):
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        
        for _ in range(layers_count):
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.BatchNormalization())  # Consistency: Add BatchNormalization
            model.add(layers.MaxPooling2D((2, 2)))
        
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
    def build_alexnet_base(self, input_shape):
        model = models.Sequential()
        
        # Layer 1: Convolutional + BatchNormalization + MaxPooling
        model.add(layers.Conv2D(96, (11, 11), strides=4, padding='same', activation='relu', input_shape=input_shape))
        model.add(layers.BatchNormalization())  # Add BatchNormalization
        model.add(layers.MaxPooling2D((3, 3), strides=2))
    
        # Layer 2: Convolutional + BatchNormalization + MaxPooling
        model.add(layers.Conv2D(256, (5, 5), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())  # Add BatchNormalization
        model.add(layers.MaxPooling2D((3, 3), strides=2))
    
        # Layer 3: Convolutional + BatchNormalization
        model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())  # Add BatchNormalization
    
        # Layer 4: Convolutional + BatchNormalization
        model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())  # Add BatchNormalization
    
        # Layer 5: Convolutional + BatchNormalization + MaxPooling
        model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())  # Add BatchNormalization
        model.add(layers.MaxPooling2D((3, 3), strides=2))
    
        return model


    def build_base_model(self, backbone_name, input_shape):
        if backbone_name == 'CNN5':
            base_model = self.build_cnn_base(input_shape=input_shape, layers_count=5)
        elif backbone_name == 'CNN3':
            # Define the CNN3 architecture directly here
            inputs = layers.Input(shape=input_shape)
            x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
            x = layers.BatchNormalization()(x)  # Adding BatchNormalization for consistency
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Conv2D(64, (3, 3), activation='relu')(x)
            x = layers.BatchNormalization()(x)  # Adding BatchNormalization for consistency
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Conv2D(128, (3, 3), activation='relu')(x)
            x = layers.BatchNormalization()(x)  # Adding BatchNormalization for consistency
            x = layers.MaxPooling2D((2, 2))(x)
            
            #x = layers.GlobalAveragePooling2D()(x)  # This is valid for CNN backbones
            base_model = models.Model(inputs, x)  # Creating the model
        elif backbone_name == 'ResNet18':
            base_model = self.build_resnet18(input_shape=input_shape)
        elif backbone_name == 'AlexNet':
            base_model = self.build_alexnet_base(input_shape=input_shape)
        elif backbone_name == 'VGG16':
            base_model = tf.keras.applications.VGG16(include_top=False, input_shape=input_shape, weights='imagenet')
        elif backbone_name == 'VGG19':
            base_model = tf.keras.applications.VGG19(include_top=False, input_shape=input_shape, weights='imagenet')
        elif backbone_name == 'ResNet50':
            base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')
        elif backbone_name == 'DenseNet121':
            base_model = tf.keras.applications.DenseNet121(include_top=False, input_shape=input_shape, weights='imagenet')
        else:
            raise ValueError(f"Unknown backbone {backbone_name}")
        
        return base_model
    


    def build_parallel_model(self, base_model):
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)

        # Texture classification head
        texture_x = layers.Dense(128, activation='relu')(x)
        texture_output = layers.Dense(len(self.texture_classes), activation='softmax', name='texture_output')(texture_x)

        # Stiffness classification head
        stiffness_x = layers.Dense(128, activation='relu')(x)
        stiffness_output = layers.Dense(len(self.stiffness_classes), activation='softmax', name='stiffness_output')(stiffness_x)

        model = models.Model(inputs=base_model.input, outputs=[texture_output, stiffness_output])
        return model

    def build_hierarchical_stiffness_first_model(self, base_model):
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)

        # Stiffness classification head
        stiffness_x = layers.Dense(128, activation='relu')(x)
        stiffness_output = layers.Dense(len(self.stiffness_classes), activation='softmax', name='stiffness_output')(stiffness_x)

        # Texture classification head (based on stiffness output)
        texture_x = layers.Dense(128, activation='relu')(stiffness_output)
        texture_output = layers.Dense(len(self.texture_classes), activation='softmax', name='texture_output')(texture_x)

        model = models.Model(inputs=base_model.input, outputs=[stiffness_output, texture_output])
        return model

    def build_hierarchical_texture_first_model(self, base_model):
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)

        # Texture classification head
        texture_x = layers.Dense(128, activation='relu')(x)
        texture_output = layers.Dense(len(self.texture_classes), activation='softmax', name='texture_output')(texture_x)

        # Stiffness classification head (based on texture output)
        stiffness_x = layers.Dense(128, activation='relu')(texture_output)
        stiffness_output = layers.Dense(len(self.stiffness_classes), activation='softmax', name='stiffness_output')(stiffness_x)

        model = models.Model(inputs=base_model.input, outputs=[texture_output, stiffness_output])
        return model

    def train_and_evaluate(self, model, model_name, data_ratio):
        train_gen, val_gen, test_gen = self.get_data_generators(data_ratio)
        self.model = model
        self.dataset_ratio = data_ratio

        model.compile(optimizer=optimizers.Adam(lr=0.0005),
                      loss={'texture_output': 'categorical_crossentropy',
                            'stiffness_output': 'categorical_crossentropy'},
                      metrics={'texture_output': 'accuracy',
                               'stiffness_output': 'accuracy'})

        # Define callbacks
        #early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        checkpoint = ModelCheckpoint(filepath=f'{model_name}_dr{data_ratio}-best_model.h5',
                                     save_best_only=True, monitor='val_loss', mode='min')
        csv_logger = CSVLogger(f'{model_name}_dr{data_ratio}-training_log.csv')

        # Train the model
        start_time = time.time()
        history = model.fit(train_gen,
                            epochs=EPOCHS,
                            validation_data=val_gen,
                            callbacks=[checkpoint, csv_logger])
        end_time = time.time()

        self.training_time = end_time - start_time
        self.actual_epochs = len(history.epoch)

        # Save the final model after training
        model.save(f'{model_name}_dr{data_ratio}-final_model.h5')

        # Evaluate and log performance
        self.evaluate_and_log(train_gen, val_gen, test_gen)

    def evaluate_and_log(self, train_gen, val_gen, test_gen):
        def evaluate_generator(generator):
            results = self.model.evaluate(generator)
            return {
                'total_loss': results[0],
                'texture_loss': results[1],
                'stiffness_loss': results[2],
                'texture_accuracy': results[3],
                'stiffness_accuracy': results[4]
            }

        # Evaluate on training, validation, and test sets
        train_metrics = evaluate_generator(train_gen)
        val_metrics = evaluate_generator(val_gen)
        test_metrics = evaluate_generator(test_gen)

        # Calculate Frames Per Second (FPS)
        start_time = time.time()
        test_samples = test_gen.samples
        self.model.evaluate(test_gen)
        end_time = time.time()
        evaluation_time = end_time - start_time
        fps = test_samples / evaluation_time

        # Get total number of parameters
        total_params = self.model.count_params()

        log_text = (
            f"Backbone: {self.backbone}, Ratio: {self.dataset_ratio}\n"
            f"Training Loss: {train_metrics['total_loss']:.4f}, Training Texture Loss: {train_metrics['texture_loss']:.4f}, Training Stiffness Loss: {train_metrics['stiffness_loss']:.4f}, "
            f"Training Texture Accuracy: {train_metrics['texture_accuracy']:.4f}, Training Stiffness Accuracy: {train_metrics['stiffness_accuracy']:.4f}\n"
            f"Validation Loss: {val_metrics['total_loss']:.4f}, Validation Texture Loss: {val_metrics['texture_loss']:.4f}, Validation Stiffness Loss: {val_metrics['stiffness_loss']:.4f}, "
            f"Validation Texture Accuracy: {val_metrics['texture_accuracy']:.4f}, Validation Stiffness Accuracy: {val_metrics['stiffness_accuracy']:.4f}\n"
            f"Test Loss: {test_metrics['total_loss']:.4f}, Test Texture Loss: {test_metrics['texture_loss']:.4f}, Test Stiffness Loss: {test_metrics['stiffness_loss']:.4f}, "
            f"Test Texture Accuracy: {test_metrics['texture_accuracy']:.4f}, Test Stiffness Accuracy: {test_metrics['stiffness_accuracy']:.4f}\n"
            f"FPS: {fps:.2f}, Total Params: {total_params}, Training Time: {self.training_time:.2f} seconds, "
            f"Epochs: {self.actual_epochs}\n"
        )
        self.write_to_file(self.logfile, log_text)


if __name__ == "__main__":
    logfile = 'comparison_log.txt'
    train_dir = './train_data'
    val_dir = './val_data'
    test_dir = './test_data'
    study = ComparisonStudy(train_dir, val_dir, test_dir, BS, logfile)

    backbones = [  'AlexNet','ResNet50','DenseNet121', 'ResNet18','CNN3', 'VGG16' ]
    data_ratios = [1, 0.8, 0.6, 0.4, 0.2]
    structures = ['Parallel_Multi_Head']
    #, 'Hierarchical_Stiffness_First', 'Hierarchical_Texture_First'
    
    for data_ratio in data_ratios:
        for backbone in backbones:
            base_model = study.build_base_model(backbone, (256, 256, 3))
            study.backbone = backbone  # Set backbone name for logging
            
            for structure in structures:
                tf.keras.backend.clear_session()
                # Generate a unique logfile name based on the current configuration
                logfile_name = f'{backbone}_{structure}_ratio{data_ratio}_log.txt'
                study.logfile = logfile_name  # Set the logfile name for the current study
                                
                if structure == 'Parallel_Multi_Head':
                    model = study.build_parallel_model(base_model)
                elif structure == 'Hierarchical_Stiffness_First':
                    model = study.build_hierarchical_stiffness_first_model(base_model)
                elif structure == 'Hierarchical_Texture_First':
                    model = study.build_hierarchical_texture_first_model(base_model)
                
                study.train_and_evaluate(model, f'{backbone}_{structure}', data_ratio)