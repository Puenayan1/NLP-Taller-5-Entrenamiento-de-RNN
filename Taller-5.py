import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dense, Dropout
import keras_tuner as kt
from sklearn.model_selection import train_test_split

# 1. Cargar y preparar los datos bilingües
df = pd.read_csv("Combined Data es-ES.csv")
df['label'] = (df['status'] == 'Anxiety').astype(int)

X = np.concatenate([df['statement'].values, df['statement_es_es'].values])
y = np.concatenate([df['label'].values, df['label'].values])
X = np.where(pd.isnull(X), '', X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

BATCH_SIZE = 64
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

VOCAB_SIZE = 10000
encoder = TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

# 2. Definir el modelo con hiperparámetros estáticos y dinámicos
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(encoder)
    
    # --- Espacio de búsqueda hiper-enfocado ---
    # Solo explorará combinaciones entre estas dimensiones
    hp_embedding_dim = hp.Int('embedding_dim', min_value=32, max_value=96, step=32)
    model.add(Embedding(input_dim=len(encoder.get_vocabulary()), 
                        output_dim=hp_embedding_dim, 
                        mask_zero=True))
    
    hp_lstm_units = hp.Choice('lstm_units', values=[32, 64])
    model.add(Bidirectional(LSTM(hp_lstm_units)))
    
    # --- Parámetros Estáticos ---
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1)) 
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'])
    return model

# 3. Keras Tuner con ciclos cortos de eliminación (Fast Search)
tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=4,         # Techo reducido de 10 a 4 épocas
                     factor=2,             # Descarta la mitad de los peores modelos en cada ronda
                     directory='rnn_tuning_logs',
                     project_name='clasificador_ansiedad_opt')

print("Iniciando búsqueda optimizada de hiperparámetros...")
# Limitamos los pasos por época temporalmente para acelerar el sondeo
tuner.search(train_dataset, validation_data=test_dataset, epochs=4)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Mejor arquitectura encontrada:\n- Embedding Dim: {best_hps.get('embedding_dim')}\n- LSTM Units: {best_hps.get('lstm_units')}")

# 4. Entrenamiento del modelo final
print("Entrenando el modelo final con la mejor configuración...")
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(train_dataset, epochs=5, validation_data=test_dataset)

# Evaluación
test_loss, test_acc = best_model.evaluate(test_dataset)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)