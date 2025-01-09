from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and dataset
try:
    model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
    data_path = 'Cleaned_Car_data.csv'  # Update the path if necessary
    car = pd.read_csv(data_path)
except Exception as e:
    print(f"Error loading resources: {e}")
    exit()

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', 
                           companies=companies, 
                           car_models=car_models, 
                           years=years, 
                           fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        driven = int(request.form.get('kilo_driven'))

        # Create input DataFrame for the model
        input_data = pd.DataFrame({
            'name': [car_model],
            'company': [company],
            'year': [year],
            'kms_driven': [driven],
            'fuel_type': [fuel_type]
        })

        # Predict the car price
        prediction = model.predict(input_data)
        price = np.round(prediction[0], 2)
        return f"Predicted Price: ₹{price}"
    except AttributeError as e:
        return f"Model compatibility error: {e}. Please ensure compatible scikit-learn versions."
    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == '__main__':
    app.run(debug=True)
