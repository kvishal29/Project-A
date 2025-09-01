import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GaussianNoise, Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import logging
from sklearn.manifold import TSNE
from itertools import product

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

class VAESampling(Layer):
    """Custom layer for VAE sampling"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAEModel(Model):
    """Custom VAE model that properly implements loss functions"""
    def __init__(self, encoder, decoder, beta=0.5, **kwargs):
        super(VAEModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction
    
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # Calculate reconstruction loss (MSE)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(data - reconstruction),
                    axis=1
                )
            )
            # Calculate KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            )
            total_loss = reconstruction_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def compile(self, optimizer, **kwargs):
        super().compile(optimizer=optimizer, loss=lambda y_true, y_pred: 0.0, **kwargs)

class JointModel(Model):
    """Final working version with proper symbolic tensor handling"""
    def __init__(self, encoder, decoder, classifier, beta=0.5, **kwargs):
        super(JointModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.beta = beta
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.classification_loss_tracker = tf.keras.metrics.Mean(name="classification_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.accuracy_tracker = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.classification_loss_tracker,
            self.kl_loss_tracker,
            self.accuracy_tracker,
        ]
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        classification = self.classifier(z)
        return {'reconstruction': reconstruction, 'classification': classification}
    
    def train_step(self, data):
        # Handle input data (x, y_dict)
        if isinstance(data, tuple):
            x, y_dict = data
            y = y_dict.get('classification')  # Extract classification labels
        else:
            x = data
            y = None
        
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            classification = self.classifier(z)
            
            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(x - reconstruction), axis=1))
            
            # Classification loss with robust tensor handling
            classification_loss = 0.0
            if y is not None:
                epsilon = tf.keras.backend.epsilon()
                classification = tf.clip_by_value(classification, epsilon, 1. - epsilon)
                classification_loss = tf.reduce_mean(
                    tf.reduce_sum(-y * tf.math.log(classification), axis=1))
            
            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            
            total_loss = (0.3 * reconstruction_loss +
                          1.0 * classification_loss +
                          self.beta * kl_loss)
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.classification_loss_tracker.update_state(classification_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        if y is not None:
            self.accuracy_tracker.update_state(y, classification)
        
        return {m.name: m.result() for m in self.metrics}
    
    def compile(self, optimizer, **kwargs):
        super().compile(
            optimizer=optimizer,
            loss={
                'reconstruction': lambda y_true, y_pred: 0.0,
                'classification': lambda y_true, y_pred: 0.0
            },
            metrics={
                'classification': ['accuracy'],
                'reconstruction': [],
            },
            **kwargs
        )

class ImprovedAdversarialDefenseFramework:
    def __init__(self, config=None):
        self.config = config or {
            'test_split': 0.3,
            'validation_split': 0.2,
            'batch_size': 32,
            'max_epochs': 12,
            'early_stopping_patience': 3,
            'regularization_strength': 0.1,
            'dropout_rate': 0.6,
            'noise_level': 0.3,
            'learning_rate': 0.0002,
            'vae_beta': 0.5,
            'dynamic_threshold': True,
            'joint_train_epochs': 30,
            'strong_attack_epsilons': [0.7, 1.0],
            'epsilon_ranges': {
                'fgsm': [0.1, 0.2, 0.3, 0.4, 0.5],
                'pgd': [0.1, 0.2, 0.3, 0.4, 0.5],
                'blackbox': [0.05, 0.1, 0.2, 0.3],
                'gan': [0.2, 0.3, 0.4, 0.5],
                'noise': [0.4, 0.6, 0.8, 1.0]
            },
            'threshold_percentiles': [60, 65, 70, 75, 80, 85, 90]  # For threshold tuning
        }
        self.results = defaultdict(lambda: defaultdict(dict))
        self.trained_models = {}
        self.feature_names = []
        self.optimal_thresholds = {}  # Store optimal thresholds per attack type and epsilon
        self.gan_generator = None  # Store trained GAN generator
        
    def load_and_preprocess_data(self, filepath="Train_Test_IoT_Modbus.csv"):
        """Load and preprocess data"""
        try:
            # Print library versions
            print("\nTensorFlow Version: " + tf.__version__)
            print("Keras Version: " + tf.keras.__version__)
            
            dataset = pd.read_csv(filepath)
            logger.info(f"Dataset loaded: {dataset.shape}")
            
            # Aggressive cleaning
            dataset = dataset.dropna().drop_duplicates()
            
            # Check the shape
            print("Shape of the input file - ", dataset.shape)
            print("\nDataset head:")
            print(dataset.head())
            
            # Information of the input data
            dataset.info()
            
            # Encode categorical features
            encoders = {}
            for col in ['date', 'time', 'type']:
                if col in dataset.columns:
                    encoder = LabelEncoder()
                    dataset[col] = encoder.fit_transform(dataset[col])
                    encoders[col] = encoder
            
            X = dataset.drop('label', axis='columns')
            y = dataset['label']
            self.feature_names = list(X.columns)
            
            # Add noise to prevent overfitting
            X_values = X.values.astype(np.float32)
            noise_scale = np.std(X_values, axis=0) * 0.08
            X_values = X_values + np.random.normal(0, noise_scale, X_values.shape)
            
            logger.info(f"Processed data: X={X_values.shape}, Classes: {Counter(y)}")
            return X_values, y.values
        except FileNotFoundError:
            logger.error(f"Dataset '{filepath}' not found!")
            raise
    
    def build_regularized_classifier(self, input_shape, name="classifier"):
        """Build classifier with heavy regularization"""
        input_layer = Input(shape=input_shape, name=f'{name}_input')
        x = GaussianNoise(self.config['noise_level'])(input_layer)
        x = Dense(24, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(self.config['regularization_strength']))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config['dropout_rate'])(x)
        x = Dense(12, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(self.config['regularization_strength']))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config['dropout_rate'])(x)
        output = Dense(2, activation='softmax', name=f'{name}_output')(x)
        
        model = Model(inputs=input_layer, outputs=output, name=name)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def build_autoencoder(self, input_size, name="autoencoder"):
        """Build enhanced Variational Autoencoder (VAE) with stronger denoising capacity"""
        # Enhanced encoder with more capacity for denoising
        encoder_input = Input(shape=(input_size,), name=f'{name}_input')
        x = GaussianNoise(0.1)(encoder_input)
        x = Dense(64, activation='relu', name='encoder_1')(x)  # Increased from 32 to 64
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu', name='encoder_2')(x)  # Additional layer
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        z_mean = Dense(16, name='z_mean')(x)  # Increased from 8 to 16
        z_log_var = Dense(16, name='z_log_var')(x)  # Increased from 8 to 16
        z = VAESampling()([z_mean, z_log_var])
        encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
        
        # Enhanced decoder with more capacity
        decoder_input = Input(shape=(16,), name='decoder_input')  # Match encoder output
        x = Dense(32, activation='relu', name='decoder_1')(decoder_input)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu', name='decoder_2')(x)  # Additional layer
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        decoder_output = Dense(input_size, activation='linear', name=f'{name}_output')(x)
        decoder = Model(decoder_input, decoder_output, name='decoder')
        
        # Full VAE model
        vae = VAEModel(encoder, decoder, beta=self.config['vae_beta'], name=name)
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        
        return vae
    
    def build_joint_model(self, input_size, name="joint_model"):
        """Build joint model with stronger separation between autoencoder and classifier"""
        # Enhanced encoder with more capacity for denoising
        encoder_input = Input(shape=(input_size,), name='reconstruction')
        x = GaussianNoise(0.1)(encoder_input)
        x = Dense(64, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.05))(x)  # Increased from 32
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(32, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.05))(x)  # Additional layer
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        z_mean = Dense(16, name='z_mean')(x)  # Increased from 8
        z_log_var = Dense(16, name='z_log_var')(x)  # Increased from 8
        z = VAESampling()([z_mean, z_log_var])
        encoder = Model(encoder_input, [z_mean, z_log_var, z], name='joint_encoder')
        
        # Enhanced reconstruction branch
        decoder_input = Input(shape=(16,), name='decoder_input')  # Match encoder output
        x = Dense(32, activation='relu')(decoder_input)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)  # Additional layer
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        decoder_output = Dense(input_size, activation='linear', name='reconstruction')(x)
        decoder = Model(decoder_input, decoder_output, name='joint_decoder')
        
        # Enhanced classification branch with stronger discrimination capacity
        classifier_input = Input(shape=(16,), name='classifier_input')  # Match encoder output
        x = Dense(32, activation='relu',  # Increased from 16
                 kernel_regularizer=tf.keras.regularizers.l2(0.05))(classifier_input)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(16, activation='relu',  # Additional layer
                 kernel_regularizer=tf.keras.regularizers.l2(0.05))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        classifier_output = Dense(2, activation='softmax', name='classification')(x)
        classifier = Model(classifier_input, classifier_output, name='joint_classifier')
        
        # Full joint model
        joint_model = JointModel(encoder, decoder, classifier, beta=self.config['vae_beta'], name=name)
        joint_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))
        
        return joint_model
    
    def train_with_overfitting_detection(self, model, X_train, y_train, name):
        """Train with strict overfitting detection"""
        logger.info(f"Training {name}...")
        
        class OverfittingCallback(tf.keras.callbacks.Callback):
            def __init__(self, patience=2, threshold=0.05):
                self.patience = patience
                self.threshold = threshold
                self.wait = 0
                self.best_val_acc = 0
            
            def on_epoch_end(self, epoch, logs=None):
                val_acc = logs.get('val_accuracy', 0)
                train_acc = logs.get('accuracy', 0)
                
                if val_acc > 0.97:
                    logger.warning(f"Stopping {self.model.name}: val_accuracy too high ({val_acc:.3f})")
                    self.model.stop_training = True
                    return
                
                if (train_acc - val_acc) > self.threshold:
                    self.wait += 1
                    if self.wait >= self.patience:
                        logger.warning(f"Stopping {self.model.name}: overfitting detected")
                        self.model.stop_training = True
                else:
                    self.wait = 0
        
        history = model.fit(
            X_train, y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['max_epochs'],
            validation_split=self.config['validation_split'],
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', mode='min',
                    patience=self.config['early_stopping_patience'],
                    restore_best_weights=True
                ),
                OverfittingCallback(patience=2, threshold=0.1)
            ],
            verbose=1
        )
        
        return model
    
    def train_autoencoder(self, X_train, model_name="autoencoder"):
        """Train VAE on both clean and adversarial examples"""
        baseline = self.trained_models['baseline']
        X_adv_fgsm = self.generate_fgsm(baseline, X_train, 0.2)
        X_adv_pgd = self.generate_pgd(baseline, X_train, 0.2)
        
        # Combine clean and adversarial data
        X_combined = np.concatenate([X_train, X_adv_fgsm, X_adv_pgd])
        y_combined = np.concatenate([X_train, X_train, X_train])  # Autoencoder target
        
        # Shuffle
        indices = np.random.permutation(len(X_combined))
        X_combined, y_combined = X_combined[indices], y_combined[indices]
        
        # Train VAE
        autoencoder = self.build_autoencoder(input_size=X_train.shape[1])
        autoencoder.fit(
            X_combined, y_combined,
            batch_size=self.config['batch_size'],
            epochs=25,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', mode='min',
                    patience=5, restore_best_weights=True
                )
            ]
        )
        
        self.trained_models[model_name] = autoencoder
    
    def build_gan_generator(self, input_size, latent_dim=8):
        """Build GAN generator for adversarial example generation"""
        model = tf.keras.Sequential([
            Dense(32, activation='relu', input_shape=(latent_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(input_size, activation='tanh')
        ], name='gan_generator')
        return model
    
    def train_gan(self, X_train, epochs=20):
        """Train GAN generator on training data"""
        input_size = X_train.shape[1]
        latent_dim = 8
        
        # Build generator and discriminator
        generator = self.build_gan_generator(input_size, latent_dim)
        discriminator = tf.keras.Sequential([
            Dense(64, activation='relu', input_shape=(input_size,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile discriminator
        discriminator.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Combined model
        discriminator.trainable = False
        gan_input = Input(shape=(latent_dim,))
        gan_output = discriminator(generator(gan_input))
        gan = Model(gan_input, gan_output)
        gan.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Training loop
        batch_size = self.config['batch_size']
        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_samples = X_train[idx]
            
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_samples = generator.predict(noise, verbose=0)
            
            d_loss_real = discriminator.train_on_batch(real_samples, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(fake_samples, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
            
            if epoch % 5 == 0:
                logger.info(f"GAN Epoch {epoch}/{epochs} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")
        
        self.gan_generator = generator
        logger.info("GAN training completed")
    
    def tune_threshold_per_attack(self, X_val, y_val):
        """Per-attack dynamic threshold tuning using grid search on validation set"""
        logger.info("Tuning thresholds per attack type...")
        
        attacks = {
            'fgsm': self.generate_fgsm,
            'pgd': self.generate_pgd,
            'blackbox': self.generate_blackbox,
            'gan': self.generate_gan,
            'noise': self.generate_noise
        }
        
        baseline = self.trained_models['baseline']
        autoencoder = self.trained_models['autoencoder']
        
        # Define grid search parameters
        percentile_grid = self.config['threshold_percentiles']
        epsilon_grid = [0.1, 0.2, 0.3, 0.4]  # Representative epsilon values
        
        best_thresholds = {}
        
        for attack_name, attack_func in attacks.items():
            logger.info(f"Tuning threshold for {attack_name}...")
            
            best_f1 = 0
            best_percentile = 70  # Default
            
            # Grid search over epsilon and percentile combinations
            for epsilon, percentile in product(epsilon_grid, percentile_grid):
                # Generate adversarial examples
                X_adv = attack_func(baseline, X_val, epsilon)
                
                # Get reconstruction errors
                z_mean, z_log_var, z = autoencoder.encoder.predict(X_adv, verbose=0)
                X_recon = autoencoder.decoder.predict(z, verbose=0)
                recon_errors = np.mean(np.abs(X_adv - X_recon), axis=1)
                
                # Apply threshold
                threshold = np.percentile(recon_errors, percentile)
                ood_mask = recon_errors > threshold
                
                # Apply denoising
                X_defended = X_val.copy()
                X_defended[ood_mask] = X_recon[ood_mask]
                
                # Get predictions
                preds = np.argmax(baseline.predict(X_defended, verbose=0), axis=1)
                y_true = np.argmax(y_val, axis=1)
                
                # Calculate F1-score
                f1 = f1_score(y_true, preds)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_percentile = percentile
            
            logger.info(f"Best threshold for {attack_name}: {best_percentile}th percentile (F1: {best_f1:.3f})")
            best_thresholds[attack_name] = best_percentile
        
        self.optimal_thresholds = best_thresholds
        return best_thresholds
    
    def train_all_models(self, X_train, y_train):
        """Complete training pipeline with enhanced joint model training"""
        logger.info("Training all defense models...")
        
        # Convert inputs to float32 numpy arrays
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train)
        
        # Convert y_train to one-hot if needed
        if len(y_train.shape) == 1:
            y_train = to_categorical(y_train)
        y_train = y_train.astype(np.float32)
        
        # Split for validation
        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
        )
        
        # 1. Baseline model
        logger.info("Training baseline classifier...")
        baseline = self.build_regularized_classifier(
            input_shape=(X_train.shape[1],), name="baseline"
        )
        self.trained_models['baseline'] = self.train_with_overfitting_detection(
            baseline, X_train_main, y_train_main, "baseline"
        )
        
        # 2. Train GAN
        logger.info("Training GAN generator...")
        self.train_gan(X_train_main)
        
        # 3. Adversarial training models
        logger.info("Training adversarial training models...")
        for i, ratio in enumerate([0.1, 0.2, 0.3], 1):
            logger.info(f"Training adversarial model {i} with {ratio*100}% adversarial data...")
            
            clean_size = int(len(X_train_main) * (1 - ratio))
            X_clean, y_clean = X_train_main[:clean_size], y_train_main[:clean_size]
            X_adv_base, y_adv = X_train_main[clean_size:], y_train_main[clean_size:]
            
            X_adv_fgsm = self.generate_fgsm(self.trained_models['baseline'], X_adv_base, 0.2)
            X_adv_pgd = self.generate_pgd(self.trained_models['baseline'], X_adv_base, 0.2)
            
            mix_size = len(X_adv_base) // 2
            X_adv_mixed = np.concatenate([X_adv_fgsm[:mix_size], X_adv_pgd[:mix_size]])
            y_adv_mixed = np.concatenate([y_adv[:mix_size], y_adv[:mix_size]])
            
            X_combined = np.concatenate([X_clean, X_adv_mixed])
            y_combined = np.concatenate([y_clean, y_adv_mixed])
            
            indices = np.random.permutation(len(X_combined))
            X_combined, y_combined = X_combined[indices], y_combined[indices]
            
            adv_model = self.build_regularized_classifier(
                input_shape=(X_train.shape[1],), name=f"adversarial_training_{i}"
            )
            self.trained_models[f'adversarial_training_{i}'] = self.train_with_overfitting_detection(
                adv_model, X_combined, y_combined, f"adversarial_training_{i}"
            )
        
        # 4. Improved autoencoder training
        self.train_autoencoder(X_train_main)
        
        # 5. Per-attack dynamic threshold tuning
        logger.info("Performing per-attack dynamic threshold tuning...")
        self.tune_threshold_per_attack(X_val, y_val)
        
        # 6. Enhanced joint model training on denoised samples
        logger.info("Training enhanced joint model on denoised samples...")
        joint_model = self.build_joint_model(input_size=X_train.shape[1])
        
        # Generate diverse adversarial examples
        baseline = self.trained_models['baseline']
        epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        attack_types = ['fgsm', 'pgd', 'blackbox', 'gan', 'noise']
        
        denoised_samples = []
        denoised_labels = []
        
        # Add clean samples
        denoised_samples.append(X_train_main)
        denoised_labels.append(y_train_main)
        
        # Generate and denoise adversarial examples
        for attack_type in attack_types:
            for epsilon in epsilon_values:
                attack_func = getattr(self, f'generate_{attack_type}')
                X_adv = attack_func(baseline, X_train_main, epsilon)
                
                # Apply OOD defense to denoise
                X_denoised, _ = self.apply_ood_defense(X_adv, attack_type, epsilon)
                
                denoised_samples.append(X_denoised)
                denoised_labels.append(y_train_main)
        
        # Combine all samples
        X_combined = np.concatenate(denoised_samples)
        y_combined = np.concatenate(denoised_labels)
        
        # Convert to tensors
        X_combined = tf.convert_to_tensor(X_combined, dtype=tf.float32)
        y_combined = tf.convert_to_tensor(y_combined, dtype=tf.float32)
        
        # Train joint model on denoised samples
        joint_model.fit(
            X_combined,
            {'classification': y_combined, 'reconstruction': X_combined},
            batch_size=self.config['batch_size'],
            epochs=self.config['joint_train_epochs'],
            validation_split=0.2,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_total_loss',
                    mode='min',
                    patience=8,
                    restore_best_weights=True
                )
            ]
        )
        
        self.trained_models['joint_model'] = joint_model
        logger.info("✓ All models trained successfully")
    
    def generate_fgsm(self, model, X, epsilon):
        """Generate FGSM attack"""
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        adv_X = fast_gradient_method(
            model, X_tensor, eps=epsilon, norm=np.inf,
            clip_min=X.min(), clip_max=X.max()
        )
        return adv_X.numpy()
    
    def generate_pgd(self, model, X, epsilon):
        """Generate PGD attack"""
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        adv_X = projected_gradient_descent(
            model, X_tensor, eps=epsilon, eps_iter=epsilon/10,
            nb_iter=20, norm=np.inf,
            clip_min=X.min(), clip_max=X.max()
        )
        return adv_X.numpy()
    
    def generate_blackbox(self, model, X, epsilon):
        """Generate black box attack"""
        noise = np.random.uniform(-epsilon, epsilon, X.shape)
        return np.clip(X + noise, X.min(), X.max())
    
    def generate_gan(self, model, X, epsilon):
        """Generate GAN attack using trained generator"""
        if self.gan_generator is None:
            logger.warning("GAN generator not trained. Using fallback method.")
            return self.generate_blackbox(model, X, epsilon)
        
        # Generate adversarial examples using GAN
        latent_dim = 8
        batch_size = X.shape[0]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gan_samples = self.gan_generator.predict(noise, verbose=0)
        
        # Scale and combine with original samples
        gan_samples = gan_samples * epsilon
        return np.clip(X + gan_samples, X.min(), X.max())
    
    def generate_noise(self, model, X, epsilon):
        """Generate noise attack"""
        noise = np.random.normal(0, epsilon, X.shape)
        return np.clip(X + noise, X.min(), X.max())
    
    def apply_ood_defense(self, X_adv, attack_type=None, epsilon=None):
        """Apply OOD defense with attack-specific thresholds"""
        # Use joint model's encoder for consistency
        joint_model = self.trained_models.get('joint_model')
        if joint_model is None:
            # Fall back to autoencoder if joint model not available
            autoencoder = self.trained_models['autoencoder']
            encoder = autoencoder.encoder
            decoder = autoencoder.decoder
        else:
            encoder = joint_model.encoder
            decoder = joint_model.decoder
        
        # Get latent representations and reconstruction
        z_mean, z_log_var, z = encoder.predict(X_adv, verbose=0)
        X_recon = decoder.predict(z, verbose=0)
        
        # Calculate reconstruction error
        recon_errors = np.mean(np.abs(X_adv - X_recon), axis=1)
        
        # Use attack-specific threshold from grid search
        if attack_type and attack_type in self.optimal_thresholds:
            threshold_percentile = self.optimal_thresholds[attack_type]
        else:
            threshold_percentile = 70  # Default
        
        threshold = np.percentile(recon_errors, threshold_percentile)
        ood_mask = recon_errors > threshold
        
        # Apply denoising
        X_defended = X_adv.copy()
        X_defended[ood_mask] = X_recon[ood_mask]
        
        detection_rate = np.mean(ood_mask)
        return X_defended, detection_rate
    
    def apply_combined_defense(self, X_adv, attack_type=None, epsilon=None):
        """Simplified combined defense without ensemble dilution"""
        try:
            # For high epsilon attacks, use pure OOD denoising
            if epsilon is not None and epsilon > 0.4:
                X_defended, detection_rate = self.apply_ood_defense(X_adv, attack_type, epsilon)
                baseline_model = self.trained_models['baseline']
                preds = baseline_model.predict(X_defended, verbose=0)
                return preds, detection_rate
            
            # For other cases, use joint model after OOD denoising
            X_defended, detection_rate = self.apply_ood_defense(X_adv, attack_type, epsilon)
            joint_model = self.trained_models['joint_model']
            z_mean, z_log_var, z = joint_model.encoder.predict(X_defended, verbose=0)
            preds = joint_model.classifier.predict(z, verbose=0)
            
            return preds, detection_rate
        except Exception as e:
            logger.error(f"Combined defense error: {str(e)}")
            # Fallback to OOD denoising
            return self.apply_ood_defense(X_adv, attack_type, epsilon)
    
    def plot_tsne(self, X, y, attack_type=None, defense_type=None, save_path=None):
        """Plot t-SNE visualization of latent space"""
        # Use autoencoder or joint model's encoder
        if 'joint_model' in self.trained_models:
            encoder = self.trained_models['joint_model'].encoder
        elif 'autoencoder' in self.trained_models:
            encoder = self.trained_models['autoencoder'].encoder
        else:
            logger.error("No encoder available for t-SNE visualization")
            return
        
        # Get latent representations
        z_mean, z_log_var, z = encoder.predict(X, verbose=0)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
        z_tsne = tsne.fit_transform(z)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Convert y to labels if one-hot encoded
        if len(y.shape) > 1:
            y_labels = np.argmax(y, axis=1)
        else:
            y_labels = y
        
        # Plot points by class
        scatter = plt.scatter(
            z_tsne[:, 0], z_tsne[:, 1], 
            c=y_labels, 
            cmap='coolwarm', 
            alpha=0.7,
            s=30
        )
        
        plt.colorbar(scatter, label='Class (0: Normal, 1: Attack)')
        
        # Add title based on context
        title = "t-SNE of Latent Space"
        if attack_type:
            title += f" - {attack_type.upper()} Attack"
        if defense_type:
            title += f" - {defense_type} Defense"
        
        plt.title(title)
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, title=None, save_path=None):
        """Plot confusion matrix for error analysis"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            cbar=False,
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack']
        )
        
        if title:
            plt.title(title)
        else:
            plt.title('Confusion Matrix')
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_tsne_plots(self, X_test, y_test):
        """Generate a single t-SNE plot showing clean, adversarial, and defended data"""
        logger.info("Generating t-SNE plot...")
        
        # Get encoder from joint model or autoencoder
        if 'joint_model' in self.trained_models:
            encoder = self.trained_models['joint_model'].encoder
        elif 'autoencoder' in self.trained_models:
            encoder = self.trained_models['autoencoder'].encoder
        else:
            logger.error("No encoder available for t-SNE visualization")
            return
        
        # Generate adversarial examples (FGSM with epsilon=0.3)
        baseline = self.trained_models['baseline']
        X_adv = self.generate_fgsm(baseline, X_test, 0.3)
        
        # Apply defense
        X_defended, _ = self.apply_ood_defense(X_adv, 'fgsm', 0.3)
        
        # Get latent representations
        z_clean = encoder.predict(X_test, verbose=0)[2]  # z is the third output
        z_adv = encoder.predict(X_adv, verbose=0)[2]
        z_defended = encoder.predict(X_defended, verbose=0)[2]
        
        # Combine all data
        all_z = np.vstack([z_clean, z_adv, z_defended])
        all_labels = np.hstack([
            np.zeros(len(X_test)),  # 0 for clean
            np.ones(len(X_adv)),    # 1 for adversarial
            np.full(len(X_defended), 2)  # 2 for defended
        ])
        
        # Get true labels for coloring
        true_labels = np.hstack([
            np.argmax(y_test, axis=1),
            np.argmax(y_test, axis=1),
            np.argmax(y_test, axis=1)
        ])
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
        z_tsne = tsne.fit_transform(all_z)
        
        # Create plot with two subplots
        plt.figure(figsize=(15, 6))
        
        # Plot by data type
        plt.subplot(1, 2, 1)
        colors = ['blue', 'red', 'green']
        labels = ['Clean', 'Adversarial', 'Defended']
        for i, (color, label) in enumerate(zip(colors, labels)):
            mask = all_labels == i
            plt.scatter(z_tsne[mask, 0], z_tsne[mask, 1], c=color, label=label, alpha=0.6)
        plt.title('t-SNE by Data Type')
        plt.legend()
        
        # Plot by true class
        plt.subplot(1, 2, 2)
        class_colors = ['purple', 'orange']
        class_labels = ['Normal', 'Attack']
        for i, (color, label) in enumerate(zip(class_colors, class_labels)):
            mask = true_labels == i
            plt.scatter(z_tsne[mask, 0], z_tsne[mask, 1], c=color, label=label, alpha=0.6)
        plt.title('t-SNE by True Class')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('tsne_latent_space.png', dpi=300)
        plt.show()
    
    def comprehensive_evaluation(self, X_test, y_test):
        """Complete evaluation with enhanced attack testing"""
        logger.info("Starting comprehensive evaluation...")
        
        attacks = {
            'fgsm': self.generate_fgsm,
            'pgd': self.generate_pgd,
            'blackbox': self.generate_blackbox,
            'gan': self.generate_gan,
            'noise': self.generate_noise
        }
        
        # Include stronger epsilon values
        for attack in ['fgsm', 'pgd']:
            self.config['epsilon_ranges'][attack].extend(self.config['strong_attack_epsilons'])
        
        defenses = ['baseline', 'adversarial_training_1', 'adversarial_training_2',
                   'adversarial_training_3', 'ood_denoising', 'combined_defense']
        
        results = []
        confusion_matrices = {}
        baseline_model = self.trained_models['baseline']
        y_true = np.argmax(y_test, axis=1)
        
        # Clean baseline evaluation
        clean_preds = np.argmax(baseline_model.predict(X_test, verbose=0), axis=1)
        clean_acc = accuracy_score(y_true, clean_preds)
        clean_f1 = f1_score(y_true, clean_preds)
        clean_precision = precision_score(y_true, clean_preds)
        clean_recall = recall_score(y_true, clean_preds)
        
        logger.info(f"Clean baseline - Accuracy: {clean_acc:.3f}, F1: {clean_f1:.3f}, Precision: {clean_precision:.3f}, Recall: {clean_recall:.3f}")
        
        for attack_name, attack_func in attacks.items():
            logger.info(f"Evaluating {attack_name.upper()}...")
            
            for epsilon in self.config['epsilon_ranges'][attack_name]:
                logger.info(f"  Testing epsilon = {epsilon}")
                
                # Generate adversarial samples
                X_adv = attack_func(baseline_model, X_test, epsilon)
                
                for defense in defenses:
                    try:
                        if defense == 'baseline':
                            pred = np.argmax(baseline_model.predict(X_adv, verbose=0), axis=1)
                            detection_rate = 0.0
                        elif defense.startswith('adversarial_training'):
                            model = self.trained_models[defense]
                            pred = np.argmax(model.predict(X_adv, verbose=0), axis=1)
                            detection_rate = 0.0
                        elif defense == 'ood_denoising':
                            X_defended, detection_rate = self.apply_ood_defense(X_adv, attack_name, epsilon)
                            pred = np.argmax(baseline_model.predict(X_defended, verbose=0), axis=1)
                        elif defense == 'combined_defense':
                            combined_preds, detection_rate = self.apply_combined_defense(X_adv, attack_name, epsilon)
                            pred = np.argmax(combined_preds, axis=1)
                        
                        # Calculate all metrics
                        accuracy = accuracy_score(y_true, pred)
                        f1 = f1_score(y_true, pred)
                        precision = precision_score(y_true, pred)
                        recall = recall_score(y_true, pred)
                        
                        results.append({
                            'Defense': defense,
                            'Attack': attack_name,
                            'Epsilon': epsilon,
                            'Accuracy': accuracy,
                            'F1': f1,
                            'Precision': precision,
                            'Recall': recall,
                            'Detection_Rate': detection_rate
                        })
                        
                        # Store confusion matrix for key scenarios only (FGSM with epsilon=0.3)
                        if attack_name == 'fgsm' and epsilon == 0.3 and defense in ['baseline', 'combined_defense']:
                            cm = confusion_matrix(y_true, pred)
                            confusion_matrices[f"{defense}_{attack_name}_eps{epsilon}"] = cm
                    
                    except Exception as e:
                        logger.error(f"Error evaluating {defense} vs {attack_name}: {str(e)}")
                        results.append({
                            'Defense': defense,
                            'Attack': attack_name,
                            'Epsilon': epsilon,
                            'Accuracy': 0.0,
                            'F1': 0.0,
                            'Precision': 0.0,
                            'Recall': 0.0,
                            'Detection_Rate': 0.0
                        })
        
        # Generate only one t-SNE plot and confusion matrices
        self.generate_tsne_plots(X_test, y_test)
        self.plot_confusion_matrices(confusion_matrices)
        
        return pd.DataFrame(results)
    
    def create_attack_vs_epsilon_plots(self, results_df):
        """Enhanced visualization of attack strength vs defense performance"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Attack Strength (ε) vs Defense Performance', fontsize=16, y=1.02)
        
        # Defense style configuration
        defense_styles = {
            'baseline': {'color': 'red', 'marker': 'o', 'linestyle': '--'},
            'adversarial_training_1': {'color': 'orange', 'marker': 's', 'linestyle': '-'},
            'adversarial_training_2': {'color': 'gold', 'marker': 'D', 'linestyle': '-'},
            'adversarial_training_3': {'color': 'green', 'marker': '^', 'linestyle': '-'},
            'ood_denoising': {'color': 'blue', 'marker': 'p', 'linestyle': ':'},
            'combined_defense': {'color': 'purple', 'marker': '*', 'linestyle': '-'}
        }
        
        attack_types = sorted(results_df['Attack'].unique())
        
        for i, attack in enumerate(attack_types):
            ax = axes[i//3, i%3] if len(attack_types) > 3 else axes[i]
            attack_data = results_df[results_df['Attack'] == attack]
            
            for defense in defense_styles.keys():
                defense_data = attack_data[attack_data['Defense'] == defense]
                if not defense_data.empty:
                    ax.plot(defense_data['Epsilon'], defense_data['F1'],
                           label=defense,
                           **defense_styles[defense])
            
            ax.set_xlabel('Attack Strength (ε)', fontsize=10)
            ax.set_ylabel('F1-Score', fontsize=10)
            ax.set_title(f'{attack.upper()} Attack', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
            
            # Add trend line for baseline
            baseline_data = attack_data[attack_data['Defense'] == 'baseline']
            if not baseline_data.empty:
                z = np.polyfit(baseline_data['Epsilon'], baseline_data['F1'], 1)
                p = np.poly1d(z)
                ax.plot(baseline_data['Epsilon'], p(baseline_data['Epsilon']),
                        color='red', alpha=0.3, linestyle='--')
        
        # Create unified legend
        legend_elements = [Line2D([0], [0],
                           color=style['color'],
                           marker=style['marker'],
                           linestyle=style['linestyle'],
                           label=defense.replace('_', ' ').title())
                          for defense, style in defense_styles.items()]
        
        fig.legend(handles=legend_elements,
                  loc='lower center',
                  ncol=len(defense_styles),
                  bbox_to_anchor=(0.5, -0.05),
                  fontsize=10)
        
        plt.tight_layout()
        plt.savefig('attack_vs_epsilon_performance-modbus.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_defense_heatmap(self, results_df):
        """Heatmap showing defense effectiveness across attacks"""
        plt.figure(figsize=(12, 8))
        
        # Calculate improvement over baseline
        heatmap_data = []
        for attack in results_df['Attack'].unique():
            attack_data = results_df[results_df['Attack'] == attack]
            baseline_mean = attack_data[attack_data['Defense'] == 'baseline']['F1'].mean()
            
            for defense in attack_data['Defense'].unique():
                if defense != 'baseline':
                    defense_mean = attack_data[attack_data['Defense'] == defense]['F1'].mean()
                    improvement = defense_mean - baseline_mean
                    heatmap_data.append({
                        'Attack': attack,
                        'Defense': defense.replace('_', ' ').title(),
                        'Improvement': improvement
                    })
        
        heatmap_df = pd.DataFrame(heatmap_data)
        pivot_table = heatmap_df.pivot(index="Defense", columns="Attack", values="Improvement")
        
        # Create heatmap using Matplotlib
        plt.imshow(pivot_table, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='F1-Score Improvement Over Baseline')
        plt.xticks(np.arange(len(pivot_table.columns)), pivot_table.columns, rotation=45)
        plt.yticks(np.arange(len(pivot_table.index)), pivot_table.index)
        plt.title('Defense Effectiveness Across Attack Types', pad=20)
        
        # Add annotations
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                plt.text(j, i, f"{pivot_table.iloc[i, j]:.2f}",
                        ha='center', va='center', color='black')
        
        plt.tight_layout()
        plt.savefig('defense_effectiveness_heatmap-modbus.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_detection_rates(self, results_df):
        """Visualize OOD detection rates across attacks"""
        plt.figure(figsize=(12, 6))
        
        # Filter for defenses with detection capability
        detection_data = results_df[
            (results_df['Defense'].isin(['ood_denoising', 'combined_defense'])) &
            (results_df['Detection_Rate'] > 0)
        ]
        
        attacks = detection_data['Attack'].unique()
        defenses = ['ood_denoising', 'combined_defense']
        colors = ['blue', 'purple']
        bar_width = 0.35
        
        for i, defense in enumerate(defenses):
            defense_data = detection_data[detection_data['Defense'] == defense]
            x = np.arange(len(attacks)) + i * bar_width
            heights = [defense_data[defense_data['Attack'] == attack]['Detection_Rate'].mean()
                      for attack in attacks]
            
            plt.bar(x, heights, bar_width, label=defense, color=colors[i])
            
            # Add value labels
            for j, height in enumerate(heights):
                plt.text(x[j], height + 0.02, f"{height:.2f}", ha='center', va='bottom')
        
        plt.xlabel('Attack Type', fontsize=12)
        plt.ylabel('Detection Rate', fontsize=12)
        plt.title('Adversarial Sample Detection Rates', fontsize=14)
        plt.xticks(np.arange(len(attacks)) + bar_width / 2, attacks)
        plt.ylim(0, 1)
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend(title='Defense Method')
        plt.tight_layout()
        plt.savefig('detection_rate_analysis-modbus.png', dpi=300)
        plt.show()
    
    def plot_confusion_matrices(self, confusion_matrices):
        """Plot confusion matrices for error analysis"""
        logger.info("Generating confusion matrices...")
        fig, axes = plt.subplots(1, len(confusion_matrices), figsize=(15, 5))
        if len(confusion_matrices) == 1:
            axes = [axes]
        
        for i, (key, cm) in enumerate(confusion_matrices.items()):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(key.replace('_', ' ').title())
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300)
        plt.show()
    
    def create_comprehensive_visualizations(self, results_df):
        """Complete visualization pipeline"""
        logger.info("Creating enhanced visualizations...")
        
        # 1. Attack vs Epsilon plots
        self.create_attack_vs_epsilon_plots(results_df)
        
        # 2. Defense effectiveness heatmap
        self.create_defense_heatmap(results_df)
        
        # 3. Detection rate analysis
        self.plot_detection_rates(results_df)
    
    def print_comprehensive_analysis(self, results_df):
        """Complete results analysis"""
        print("\n" + "="*120)
        print("COMPREHENSIVE ADVERSARIAL DEFENSE EVALUATION - FINAL RESULTS")
        print("="*120)
        
        # Overall statistics
        print(f"\nTotal evaluations performed: {len(results_df)}")
        print(f"Attacks tested: {', '.join(results_df['Attack'].unique())}")
        print(f"Defense mechanisms: {len(results_df['Defense'].unique())}")
        
        # Create pivot table for better readability
        pivot_accuracy = results_df.pivot_table(
            index=['Attack', 'Epsilon'],
            columns='Defense',
            values='F1',
            aggfunc='mean'
        )
        
        print("\n" + "="*100)
        print("F1-SCORE RESULTS TABLE")
        print("="*100)
        print(pivot_accuracy.round(3).to_string())
        
        # Find meaningful attack scenarios (where baseline F1 < 0.7)
        meaningful_results = results_df[
            (results_df['Defense'] == 'baseline') &
            (results_df['F1'] < 0.7)
        ]
        
        if not meaningful_results.empty:
            print("\n" + "="*80)
            print("DEFENSE EFFECTIVENESS ANALYSIS (Meaningful Attack Scenarios)")
            print("="*80)
            
            meaningful_attacks = meaningful_results[['Attack', 'Epsilon']].to_records(index=False)
            for attack, epsilon in meaningful_attacks:
                scenario_data = results_df[
                    (results_df['Attack'] == attack) &
                    (results_df['Epsilon'] == epsilon)
                ]
                
                print(f"\n{attack.upper()} Attack at ε={epsilon}:")
                scenario_summary = scenario_data.set_index('Defense')[['F1', 'Precision', 'Recall']].round(3)
                print(scenario_summary.to_string())
                
                # Find best defense for this scenario
                best_defense = scenario_data.loc[scenario_data['F1'].idxmax()]
                baseline_f1 = scenario_data[scenario_data['Defense'] == 'baseline']['F1'].iloc[0]
                improvement = best_defense['F1'] - baseline_f1
                
                print(f"Best Defense: {best_defense['Defense']} (F1: {best_defense['F1']:.3f}, "
                      f"Improvement: {improvement:+.3f}")
        else:
            print("\n⚠️  No meaningful attack scenarios found (all baseline F1-scores > 0.7)")
            print("Consider using higher epsilon values or more aggressive attacks.")
        
        # Defense ranking
        print("\n" + "="*80)
        print("OVERALL DEFENSE RANKING")
        print("="*80)
        
        defense_performance = results_df.groupby('Defense')['F1'].agg(['mean', 'std']).round(3)
        defense_performance = defense_performance.sort_values('mean', ascending=False)
        
        print("Defense Performance (sorted by mean F1-score):")
        print(defense_performance.to_string())
        
        # Improvement over baseline
        baseline_mean = results_df[results_df['Defense'] == 'baseline']['F1'].mean()
        print(f"\nImprovement over baseline (mean F1-score: {baseline_mean:.3f}):")
        
        for defense in ['adversarial_training_3', 'ood_denoising', 'combined_defense']:
            if defense in results_df['Defense'].values:
                defense_mean = results_df[results_df['Defense'] == defense]['F1'].mean()
                improvement = defense_mean - baseline_mean
                print(f"  {defense.replace('_', ' ').title()}: {improvement:+.3f}")

def main_improved_evaluation():
    """Complete evaluation pipeline"""
    framework = ImprovedAdversarialDefenseFramework()
    
    try:
        # Load and preprocess data
        X, y = framework.load_and_preprocess_data()
        
        # Split and normalize
        y_categorical = to_categorical(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=framework.config['test_split'],
            random_state=RANDOM_STATE, stratify=y_categorical
        )
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"Dataset: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}")
        
        # Train all models
        framework.train_all_models(X_train_scaled, y_train)
        
        # Comprehensive evaluation
        results_df = framework.comprehensive_evaluation(X_test_scaled, y_test)
        
        # Create visualizations
        framework.create_comprehensive_visualizations(results_df)
        
        # Print analysis
        framework.print_comprehensive_analysis(results_df)
        
        # Save results
        results_df.to_csv('final_comprehensive_defense_results-modbus.csv', index=False)
        logger.info("✓ Results saved to 'final_comprehensive_defense_results-modbus.csv'")
        
        return framework, results_df
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    framework, results = main_improved_evaluation()