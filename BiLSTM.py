import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,GRU, Dense, Bidirectional, TimeDistributed
from keras import backend as K
from sklearn.model_selection import train_test_split
import ast
import matplotlib.pyplot as plt

df = pd.read_csv('disease_paths_with_words_急性非ST段抬高.csv')
row_1 = df.iloc[0]
unique_item = len(ast.literal_eval(row_1['Word_Encoded_List'])[0])
print(unique_item)
# Set the ratio of training set to validation set
# 80% of the data is used for training and 20% is used for validation
train_size = 0.8
indices = df.index.tolist()
# Split the data set into training set and other parts (validation set + test set)
train_indices, temp_indices = train_test_split(indices, train_size=train_size, random_state=42)
# Divide half of the remaining 40% of the data as the validation set and half as the test set
val_size = 0.5
val_indices, test_indices = train_test_split(temp_indices, test_size=val_size, random_state=42)
#Evaluation index
def accuracy(y_true, y_pred):
    correct_predictions = K.equal(K.round(y_true), K.round(y_pred))
    accuracy = K.mean(K.cast(correct_predictions, tf.float32))
    return accuracy
def precision_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def f1_metric(y_true, y_pred):
    p = precision_metric(y_true, y_pred)
    r = recall_metric(y_true, y_pred)
    f1_val = 2 * (p * r) / (p + r + K.epsilon())
    return f1_val

# Set model input and output data dimensions
input_dimension = unique_item
output_dimension = unique_item
# Model building
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(None, input_dimension)))
model.add(TimeDistributed(Dense(output_dimension, activation='sigmoid')))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[accuracy, precision_metric, recall_metric, f1_metric])
# Initialize the dictionary that stores training history
history_dict = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

# Model Training
for index in train_indices:
    row = df.loc[index]
    encoded_list = eval(row['Word_Encoded_List'])
# Traverse the data row by row, training for each disease course individually
    X = []
    Y = []
    for i in range(len(encoded_list) - 1):
        X.append(encoded_list[i])
        Y.append(encoded_list[i + 1])
    X = np.array(X)
    Y = np.array(Y)
    X = np.reshape(X, (1, X.shape[0], X.shape[1]))
    Y = np.reshape(Y, (1, Y.shape[0], Y.shape[1]))
    history = model.fit(X, Y, batch_size=1, epochs=1, verbose=1)
    # Store training model evaluation results
    history_dict['loss'].append(history.history['loss'][0])
    history_dict['accuracy'].append(history.history['accuracy'][0])
    history_dict['precision'].append(history.history['precision_metric'][0])
    history_dict['recall'].append(history.history['recall_metric'][0])
    history_dict['f1_score'].append(history.history['f1_metric'][0])
    # Reset model state
    model.reset_states()
# Draw evaluation index curve
epochs = range(1, len(history_dict['loss']) + 1)
plt.figure(figsize=(14, 5))
plt.subplot(1, 5, 1)
plt.plot(epochs, history_dict['loss'], 'bo-', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 5, 2)
plt.plot(epochs, history_dict['accuracy'], 'ro-', label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 5, 3)
plt.plot(epochs, history_dict['precision'], 'go-', label='Training Precision')
plt.title('Training Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()

plt.subplot(1, 5, 4)
plt.plot(epochs, history_dict['recall'], 'mo-', label='Training Recall')
plt.title('Training Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()

plt.subplot(1, 5, 5)
plt.plot(epochs, history_dict['f1_score'], 'co-', label='Training F1 Score')
plt.title('Training F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.tight_layout()
plt.show()

# Model validation
val_accuracy_scores = []
val_precision_scores = []
val_recall_scores = []
val_f1_scores = []

for index in val_indices:
    row = df.loc[index]
    encoded_list = eval(row['Word_Encoded_List'])
    X_val = []
    Y_val = []
    for i in range(len(encoded_list) - 1):
        X_val.append(encoded_list[i])
        Y_val.append(encoded_list[i + 1])
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    X_val = np.reshape(X_val, (1, X_val.shape[0], X_val.shape[1]))  # 批量大小为1
    Y_val = np.reshape(Y_val, (1, Y_val.shape[0], Y_val.shape[1]))  # 批量大小为1
    Y_pred = model.predict(X_val, batch_size=1)
    Y_val_casted = K.cast(Y_val, 'float32')
    val_accuracy_scores.append(K.eval(accuracy(Y_val_casted, Y_pred)))
    val_precision_scores.append(K.eval(precision_metric(Y_val_casted, Y_pred)))
    val_recall_scores.append(K.eval(recall_metric(Y_val_casted, Y_pred)))
    val_f1_scores.append(K.eval(f1_metric(Y_val_casted, Y_pred)))


epochs = range(1, len(val_accuracy_scores) + 1)
plt.figure(figsize=(14, 5))

plt.subplot(1, 4, 1)
plt.plot(epochs, val_accuracy_scores, 'bo-', label='Val Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 4, 2)
plt.plot(epochs, val_precision_scores, 'ro-', label='Val Precision')
plt.title('Validation Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()

plt.subplot(1, 4, 3)
plt.plot(epochs, val_recall_scores, 'go-', label='Val Recall')
plt.title('Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()

plt.subplot(1, 4, 4)
plt.plot(epochs, val_f1_scores, 'mo-', label='Val F1 Score')
plt.title('Validation F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()

# Model prediction
eval_accuracy = []
eval_precision = []
eval_recall = []
eval_f1 = []

process_accuracy = []
process_precision = []
process_recall = []
process_f1 = []
for index in test_indices:
    row = df.loc[index]
    encoded_list = eval(row['Word_Encoded_List'])

    # Input the data of the first diagnosis and treatment day
    X_test = np.array([encoded_list[0]])
    X_test = np.reshape(X_test, (1, 1, input_dimension))

    # predict the complete diagnosis and treatment pattern in sequence
    predicted_sequence = []
    for i in range(len(encoded_list) - 1):
        # Predict the next treatment day
        Y_pred = model.predict(X_test, batch_size=1)
        predicted_sequence.append(Y_pred[0, 0, :])
        # Update input to predicted output
        X_test = np.reshape(Y_pred, (1, 1, input_dimension))

    Y_pred_sequence = np.array(predicted_sequence)
    Y_true_sequence = np.array(encoded_list[1:])

    process_accuracy.append(accuracy(K.variable(Y_true_sequence), K.variable(Y_pred_sequence)).numpy())
    process_precision.append(precision_metric(K.variable(Y_true_sequence), K.variable(Y_pred_sequence)).numpy())
    process_recall.append(recall_metric(K.variable(Y_true_sequence), K.variable(Y_pred_sequence)).numpy())
    process_f1.append(f1_metric(K.variable(Y_true_sequence), K.variable(Y_pred_sequence)).numpy())

    eval_accuracy.append(accuracy(K.variable(Y_true_sequence), K.variable(Y_pred_sequence)).numpy())
    eval_precision.append(precision_metric(K.variable(Y_true_sequence), K.variable(Y_pred_sequence)).numpy())
    eval_recall.append(recall_metric(K.variable(Y_true_sequence), K.variable(Y_pred_sequence)).numpy())
    eval_f1.append(f1_metric(K.variable(Y_true_sequence), K.variable(Y_pred_sequence)).numpy())

# The average of the prediction result evaluation indicators
mean_accuracy = np.mean(eval_accuracy)
mean_precision = np.mean(eval_precision)
mean_recall = np.mean(eval_recall)
mean_f1 = np.mean(eval_f1)

print(f'Mean Accuracy: {mean_accuracy}')
print(f'Mean Precision: {mean_precision}')
print(f'Mean Recall: {mean_recall}')
print(f'Mean F1 Score: {mean_f1}')

# Draw graphs
plt.figure(figsize=(14, 8))

plt.subplot(1, 4, 1)
plt.plot(process_accuracy, 'bo-', label='Accuracy')
plt.title('Accuracy per Process')
plt.xlabel('Process Index')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 4, 2)
plt.plot(process_precision,  'ro-',label='Precision')
plt.title('Precision per Process')
plt.xlabel('Process Index')
plt.ylabel('Precision')
plt.legend()

plt.subplot(1, 4, 3)
plt.plot(process_recall, 'go-', label='Recall')
plt.title('Recall per Process')
plt.xlabel('Process Index')
plt.ylabel('Recall')
plt.legend()

plt.subplot(1, 4, 4)
plt.plot(process_f1, 'mo-', label='F1 Score')
plt.title('F1 Score per Process')
plt.xlabel('Process Index')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()






