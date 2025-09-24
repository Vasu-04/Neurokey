import sys
import json
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def main():
    try:
        # Read input from Node.js
        input_data = json.loads(sys.stdin.read().strip())
        test_sample = input_data.get('input', {})
        # print("Received input:", test_sample, file=sys.stderr)
        # csv_file = input_data.get('csv_file', 'keystroke_dataset.csv')
        csv_file = "C:\\Users\\Aditya Jindal\\OneDrive\\Desktop\\College Data\\Minor Project\\Neurokey\\backend\\src\\Python\\keystrokeData.csv"
        threshold = 95  # Default threshold percentage
        # Validate test sample
        # if not all(key in test_sample for key in ['dwell', 'flight', 'interkey']):
        #     raise ValueError("test_sample must contain dwell, flight, and interkey values")
        # print("holla")
        # Load dataset from CSV
        try:
            df = pd.read_csv(csv_file)
            # print("dataset :", df.head())
        except FileNotFoundError:
            raise ValueError(f"Dataset file '{csv_file}' not found")
        
        # Validate dataset
        # required_columns = ['dwell', 'flight', 'interkey', 'target']
        # if not all(col in df.columns for col in required_columns):
        #     raise ValueError(f"Dataset must contain columns: {required_columns}")
        
        # if len(df) == 0:
        #     raise ValueError("Dataset is empty")
        
        # Separate features and target
        X = df[['dwell', 'flight', 'interkey']].values
        y = df['target'].values
        
        # Handle case with only one sample (add small noise for Naive Bayes)
        if len(df) == 1:
            # Create synthetic samples by adding small gaussian noise
            original_sample = X[0]
            original_target = y[0]
            
            # Generate 3 additional samples with small variations
            noise_samples = []
            noise_targets = []
            
            for i in range(3):
                # Add 1-5% random noise to each feature
                noise = np.random.normal(0, 0.03, 3)  # 3% standard deviation
                noisy_sample = original_sample * (1 + noise)
                noise_samples.append(noisy_sample)
                noise_targets.append(original_target)
            
            # Combine original with noisy samples
            X = np.vstack([X] + noise_samples)
            y = np.concatenate([y] + [np.array(noise_targets)])
            
        # Scale features for better performance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Gaussian Naive Bayes classifier
        # Naive Bayes works well with small datasets and provides probabilistic output
        classifier = GaussianNB()
        classifier.fit(X_scaled, y)
        
        # Prepare test sample
        test_array = np.array([[test_sample['dwellTime'], test_sample['flightTime'], test_sample['interkeyTime']]])
        test_scaled = scaler.transform(test_array)
        
        # Make prediction
        # prediction = classifier.predict(test_scaled)[0]
        probabilities = classifier.predict_proba(test_scaled)[0]
        
        # Get probability for actual user (class 1)
        if 1 in classifier.classes_:
            idx = np.where(classifier.classes_ == 1)[0][0]
            actual_user_prob = probabilities[idx] * 100
        else:
            actual_user_prob = 0.0
        # imposter_prob = probabilities[0] * 100
        print("Actual User Probability:", actual_user_prob, file=sys.stderr)
        # Classify based on threshold
        is_actual_user = bool(actual_user_prob >= threshold)

        # classification = "Actual User" if is_actual_user else "Imposter"
        
        # Determine confidence level
        # confidence_level = determine_confidence_level(actual_user_prob)
        
        # Calculate additional metrics
        # dataset_stats = calculate_dataset_stats(df)
        
        # Prepare response
        response = {
            # 'status': 'success',
            # 'input': test_sample,
            # 'prediction': int(prediction),
            'actual_probability': round(actual_user_prob, 2),
            # 'imposter_probability': round(imposter_prob, 2),
            # 'threshold': threshold,
            # 'classification': classification,
            # 'confidence_level': confidence_level,
            'authenticated': is_actual_user,
            # 'algorithm': 'Gaussian Naive Bayes',
            # 'dataset_info': {
            #     'total_samples': len(df),
            #     'actual_users': int(np.sum(df['target'] == 1)),
            #     'imposters': int(np.sum(df['target'] == 0)),
            #     'stats': dataset_stats
            # }
        }
        
        print(json.dumps(response))
        # print("at last of the program")
    except Exception as e:
        error_response = {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }
        print(json.dumps(error_response))
        sys.exit(1)



if __name__ == "__main__":
    main()