from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load model
model_pipeline = joblib.load('churn_prediction_pipeline.joblib')
model = model_pipeline['model']
columns_info = joblib.load('churn_columns.joblib')
numerical_features = columns_info['numerical_cols']
categorical_features = columns_info['categorical_cols']

print(f"Model loaded. {len(numerical_features)} numerical + {len(categorical_features)} categorical features")

class SafePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def transform(self, X):
        # Use the ORIGINAL trained pipeline's transform method directly
        return self.pipeline['preprocessor'].transform(X)

# Use original preprocessor safely
preprocessor = SafePreprocessor(model_pipeline)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Extract form data
            form_data = request.form.to_dict()
            
            # Create data dictionary with correct types
            customer_data = {}
            for feature in numerical_features + categorical_features:
                if feature in numerical_features:
                    customer_data[feature] = float(form_data.get(feature, 0))
                else:
                    customer_data[feature] = form_data.get(feature, categorical_features[0])
            
            # Create DataFrame with EXACT column order
            df = pd.DataFrame([customer_data])[numerical_features + categorical_features]
            
            # Transform using ORIGINAL trained preprocessor
            X_transformed = model_pipeline['preprocessor'].transform(df)
            
            # Predict
            prediction = model.predict(X_transformed)[0]
            probabilities = model.predict_proba(X_transformed)[0]
            
            result = {
                'prediction': 'CHURN' if prediction == 1 else 'STAY',
                'churn_probability': f"{probabilities[1]:.1%}",
                'confidence': f"{max(probabilities):.1%}",
                'risk_level': 'high' if probabilities[1] > 0.7 else 'medium' if probabilities[1] > 0.4 else 'low'
            }
            
            return render_template_string(TEMPLATE, result=result)
            
        except Exception as e:
            return f"<h1 style='color:red;padding:40px'>Error: {str(e)}</h1><a href='/' style='display:block;margin:20px auto;width:200px;padding:10px;background:#4a90e2;color:white;text-align:center;text-decoration:none;border-radius:5px'>Try Again</a>"
    
    return render_template_string(TEMPLATE)

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Churn Prediction System</title>

    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #4a90e2 100%);
            min-height: 100vh; padding: 20px 0;
        }
        .container { 
            max-width: 1200px; margin: 0 auto; 
            background: rgba(255,255,255,0.95); 
            border-radius: 20px; 
            box-shadow: 0 20px 60px rgba(0,0,0,0.3); 
            padding: 40px;
            backdrop-filter: blur(10px);
        }
        .header { text-align: center; margin-bottom: 40px; }
        h1 { color: #1e3c72; font-size: 2.8rem; margin-bottom: 10px; }
        .subtitle { color: #4a90e2; font-size: 1.2rem; font-weight: 500; }
        
        .form-section { margin-bottom: 30px; }
        .section-title { 
            color: #1e3c72; 
            font-size: 1.4rem; 
            margin-bottom: 20px; 
            padding-bottom: 10px; 
            border-bottom: 3px solid #4a90e2;
        }
        
        .form-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 25px; 
            margin-bottom: 20px;
        }
        .form-grid-2cols { grid-template-columns: repeat(2, 1fr); }
        .form-group { 
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            padding: 25px; 
            border-radius: 15px; 
            border: 2px solid rgba(74,144,226,0.3);
            transition: all 0.3s ease;
        }
        .form-group:hover { 
            transform: translateY(-5px); 
            box-shadow: 0 15px 30px rgba(74,144,226,0.4);
        }
        label { 
            display: block; font-weight: 600; margin-bottom: 10px; 
            color: #1e3c72; font-size: 1rem;
        }
        input, select { 
            width: 100%; padding: 15px; border: 2px solid #bbdefb; 
            border-radius: 10px; font-size: 16px; background: white;
            transition: all 0.3s ease;
        }
        input:focus, select:focus { 
            border-color: #4a90e2; outline: none; 
            box-shadow: 0 0 0 4px rgba(74,144,226,0.2);
        }
        
        .predict-button { 
            background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
            color: white; padding: 20px 60px; border: none; 
            border-radius: 50px; font-size: 1.4rem; font-weight: bold; 
            cursor: pointer; width: 100%; margin-top: 30px;
            transition: all 0.3s ease; text-transform: uppercase;
            letter-spacing: 1px;
        }
        .predict-button:hover { 
            transform: translateY(-5px); 
            box-shadow: 0 20px 40px rgba(74,144,226,0.5);
        }
        
        .result { 
            padding: 40px; border-radius: 20px; margin: 40px 0; 
            text-align: center; font-size: 1.3rem;
        }
        .result.high { background: linear-gradient(135deg, #ff6b6b, #ee5a52); color: white; }
        .result.medium { background: linear-gradient(135deg, #feca57, #ff9ff3); color: #2c3e50; }
        .result.low { background: linear-gradient(135deg, #00b894, #00cec9); color: white; }
        
        .metrics-grid { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 30px; margin: 30px 0;
        }
        .metric-card { 
            background: rgba(255,255,255,0.9); padding: 30px; 
            border-radius: 15px; text-align: center;
        }
        .metric-value { font-size: 2.5rem; font-weight: bold; color: #1e3c72; margin-bottom: 10px; }
        .metric-label { color: #5a6c7d; font-size: 1rem; font-weight: 500; }
        
        .recommendations { 
            background: rgba(255,255,255,0.9); padding: 25px; 
            border-radius: 12px; margin-top: 25px; text-align: left;
        }
        .recommendations h4 { color: #1e3c72; margin-bottom: 15px; }
        .recommendations ul { margin: 10px 0; padding-left: 25px; }
        .recommendations li { margin: 10px 0; font-size: 1.1rem; }
        
        @media (max-width: 768px) {
            .form-grid, .form-grid-2cols { grid-template-columns: 1fr; }
            .container { margin: 10px; padding: 20px; }
            h1 { font-size: 2rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Churn Prediction System</h1>
            <p class="subtitle">97.8% Accurate Production Deployment</p>
        </div>

        <form method="POST">
            <div class="form-section">
                <h3 class="section-title">Customer Profile</h3>
                <div class="form-grid form-grid-2cols">
                    <div class="form-group">
                        <label>Tenure (months)</label>
                        <input type="number" name="Tenure" value="12" min="0" max="60" step="0.1" required>
                    </div>
                    <div class="form-group">
                        <label>City Tier (1-3)</label>
                        <input type="number" name="CityTier" value="1" min="1" max="3" required>
                    </div>
                    <div class="form-group">
                        <label>Warehouse to Home Distance (km)</label>
                        <input type="number" name="WarehouseToHome" value="15" min="5" max="50" step="0.1" required>
                    </div>
                    <div class="form-group">
                        <label>Number of Addresses</label>
                        <input type="number" name="NumberOfAddress" value="3" min="1" max="20" required>
                    </div>
                </div>
            </div>

            <div class="form-section">
                <h3 class="section-title">App Usage & Satisfaction</h3>
                <div class="form-grid">
                    <div class="form-group">
                        <label>Hours Spent on App (daily)</label>
                        <input type="number" name="HourSpendOnApp" value="3" max="5" step="0.1" required>
                    </div>
                    <div class="form-group">
                        <label>Devices Registered</label>
                        <input type="number" name="NumberOfDeviceRegistered" value="3" min="1" max="6" required>
                    </div>
                    <div class="form-group">
                        <label>Satisfaction Score (1-5)</label>
                        <input type="number" name="SatisfactionScore" value="3" min="1" max="5" required>
                    </div>
                    <div class="form-group">
                        <label>Complaints (0=No, 1=Yes)</label>
                        <input type="number" name="Complain" value="0" min="0" max="1" required>
                    </div>
                </div>
            </div>

            <div class="form-section">
                <h3 class="section-title">Order & Financial Data</h3>
                <div class="form-grid">
                    <div class="form-group">
                        <label>Order Amount Growth (%)</label>
                        <input type="number" name="OrderAmountHikeFromlastYear" value="15" min="10" max="30" step="0.1" required>
                    </div>
                    <div class="form-group">
                        <label>Coupons Used</label>
                        <input type="number" name="CouponUsed" value="2" min="0" max="10" required>
                    </div>
                    <div class="form-group">
                        <label>Total Order Count</label>
                        <input type="number" name="OrderCount" value="3" min="1" max="15" required>
                    </div>
                    <div class="form-group">
                        <label>Days Since Last Order</label>
                        <input type="number" name="DaySinceLastOrder" value="5" min="0" max="30" required>
                    </div>
                    <div class="form-group">
                        <label>Total Cashback Amount</label>
                        <input type="number" name="CashbackAmount" value="150" min="0" max="500" step="0.1" required>
                    </div>
                </div>
            </div>

            <div class="form-section">
                <h3 class="section-title">Customer Preferences</h3>
                <div class="form-grid form-grid-2cols">
                    <div class="form-group">
                        <label>Preferred Login Device</label>
                        <select name="PreferredLoginDevice" required>
                            <option value="Phone">Phone</option>
                            <option value="Mobile Phone">Mobile Phone</option>
                            <option value="Computer">Computer</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Preferred Payment Mode</label>
                        <select name="PreferredPaymentMode" required>
                            <option value="Debit Card">Debit Card</option>
                            <option value="UPI">UPI</option>
                            <option value="CC">CC</option>
                            <option value="Credit Card">Credit Card</option>
                            <option value="E wallet">E wallet</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Gender</label>
                        <select name="Gender" required>
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Preferred Order Category</label>
                        <select name="PreferedOrderCat" required>
                            <option value="Fashion">Fashion</option>
                            <option value="Mobile">Mobile</option>
                            <option value="Laptop & Accessory">Laptop & Accessory</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Marital Status</label>
                        <select name="MaritalStatus" required>
                            <option value="Single">Single</option>
                            <option value="Married">Married</option>
                        </select>
                    </div>
                </div>
            </div>

            <button type="submit" class="predict-button">Predict Churn Risk</button>
        </form>

        {% if result %}
        <div class="result {{ result.risk_level }}">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{{ result.prediction }}</div>
                    <div class="metric-label">Prediction</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ result.churn_probability }}</div>
                    <div class="metric-label">Churn Probability</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ result.confidence }}</div>
                    <div class="metric-label">Model Confidence</div>
                </div>
            </div>

            {% if result.prediction == 'CHURN' %}
                <h3 style="color: white; margin: 30px 0 20px 0; font-size: 1.8rem;">High Priority - Immediate Action Required</h3>
                <div class="recommendations">
                    <h4>Retention Actions:</h4>
                    <ul>
                        <li>Send 25% discount coupon immediately</li>
                        <li>Contact via priority customer support</li>
                        <li>Launch personalized retention campaign</li>
                    </ul>
                </div>
            {% else %}
                <h3 style="color: white; margin: 30px 0 20px 0; font-size: 1.8rem;">Low Risk Customer</h3>
                <div class="recommendations">
                    <h4>Proactive Retention Actions:</h4>
                    <ul>
                        <li>Award loyalty points bonus</li>
                        <li>Offer exclusive early access deals</li>
                        <li>Send birthday discount voucher</li>
                    </ul>
                </div>
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
