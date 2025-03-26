# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load and explore the data
def load_data():
    data = pd.read_csv('BankChurners.csv')
    print("Data loaded successfully!")
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nData info:")
    print(data.info())
    return data

# 2. Preprocess the data
def preprocess_data(data):
    # Encode target variable
    le = LabelEncoder()
    data['Attrition_Flag'] = le.fit_transform(data['Attrition_Flag'])
    
    # One-hot encode categorical variables
    data = pd.get_dummies(data, columns=['Gender', 'Education_Level', 'Marital_Status', 
                                        'Income_Category', 'Card_Category'], drop_first=True)
    
    # Drop unnecessary columns
    data = data.drop(['CLIENTNUM', 
                     'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                     'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], axis=1)
    
    return data

# 3. Build and train the model
def build_and_train_model(X_train, X_test, y_train, y_test):
    model = Sequential()
    
    # Input layer and first hidden layer
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.2))
    
    # Second hidden layer
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test),
                        epochs=100, 
                        batch_size=32,
                        callbacks=[early_stop],
                        verbose=1)
    
    return model, history

# 4. Evaluate the model
def evaluate_model(model, X_test, y_test, history):
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Evaluate on test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'\nTest Accuracy: {accuracy*100:.2f}%')
    
    # Classification report
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

# 5. Feature importance
def show_feature_importance(model, X):
    weights = model.layers[0].get_weights()[0]
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(weights).mean(axis=1)
    }).sort_values('Importance', ascending=False)
    
    print('\nTop 10 Important Features:')
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
    plt.title('Top 10 Important Features')
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.show()

# Main execution
def main():
    # Step 1: Load data
    data = load_data()
    
    # Step 2: Preprocess data
    processed_data = preprocess_data(data)
    
    # Split into features and target
    X = processed_data.drop('Attrition_Flag', axis=1)
    y = processed_data['Attrition_Flag']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Step 3: Build and train model
    model, history = build_and_train_model(X_train, X_test, y_train, y_test)
    
    # Step 4: Evaluate model
    evaluate_model(model, X_test, y_test, history)
    
    # Step 5: Show feature importance
    show_feature_importance(model, X)

if __name__ == "__main__":
    main()