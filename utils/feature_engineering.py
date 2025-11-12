import pandas as pd

def engineer_features(df):
    """
    Calculate derived features from user input
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Calculate expenses based on salary
    df['school_fees'] = df['monthly_salary'] * 0.05
    df['college_fees'] = df['monthly_salary'] * 0.03
    df['travel_expenses'] = df['monthly_salary'] * 0.02
    df['groceries_utilities'] = df['monthly_salary'] * 0.15
    df['other_monthly_expenses'] = df['monthly_salary'] * 0.08
    
    # Assume default values for fields not collected
    df['family_size'] = 4
    df['dependents'] = 2
    df['existing_loans'] = 0
    df['current_emi_amount'] = 0
    
    # Calculate loan request defaults
    df['requested_amount'] = df['monthly_salary'] * 12
    df['requested_tenure'] = 24
    
    # Calculate financial ratios
    df['total_monthly_expenses'] = (
        df['monthly_rent'] + 
        df['school_fees'] + 
        df['college_fees'] + 
        df['travel_expenses'] + 
        df['groceries_utilities'] + 
        df['other_monthly_expenses']
    )
    
    df['savings_capacity'] = df['monthly_salary'] - df['total_monthly_expenses']
    df['max_monthly_emi'] = df['savings_capacity'] * 0.4
    
    # Handle division by zero
    df['debt_to_income_ratio'] = df['current_emi_amount'] / df['monthly_salary'].replace(0, 1)
    df['financial_stability'] = df['bank_balance'] / df['monthly_salary'].replace(0, 1)
    df['per_capita_income'] = df['monthly_salary'] / df['family_size']
    df['employment_stability'] = df['years_of_employment'] / 10
    df['housing_burden_ratio'] = df['monthly_rent'] / df['monthly_salary'].replace(0, 1)
    df['loan_to_income_ratio'] = df['requested_amount'] / (df['monthly_salary'].replace(0, 1) * 12)
    df['expance_to_income_ratio'] = df['total_monthly_expenses'] / df['monthly_salary'].replace(0, 1)
    df['affordability_ratio'] = df['max_monthly_emi'] / df['monthly_salary'].replace(0, 1)
    
    df['emi_scenario'] = 1
    
    # One-hot encoding for categorical variables - Initialize all to 0 first
    df['education_high school'] = 0
    df['education_post graduate'] = 0
    df['education_professional'] = 0
    
    df['employment_type_private'] = 0
    df['employment_type_self-employed'] = 0
    
    df['company_type_mid-size'] = 0
    df['company_type_mnc'] = 0
    df['company_type_small'] = 0
    df['company_type_startup'] = 0
    
    df['house_type_own'] = 0
    df['house_type_rented'] = 0
    
    # Set the appropriate ones to 1
    education_val = df['education'].iloc[0]
    if education_val == 'High School':
        df['education_high school'] = 1
    elif education_val == 'Post Graduate':
        df['education_post graduate'] = 1
    elif education_val == 'Professional':
        df['education_professional'] = 1
    
    employment_val = df['employment_type'].iloc[0]
    if employment_val == 'Private':
        df['employment_type_private'] = 1
    elif employment_val == 'Self-Employed':
        df['employment_type_self-employed'] = 1
    
    company_val = df['company_type'].iloc[0]
    if company_val == 'Mid-Size':
        df['company_type_mid-size'] = 1
    elif company_val == 'MNC':
        df['company_type_mnc'] = 1
    elif company_val == 'Small':
        df['company_type_small'] = 1
    elif company_val == 'Startup':
        df['company_type_startup'] = 1
    
    house_val = df['house_type'].iloc[0]
    if house_val == 'Own':
        df['house_type_own'] = 1
    elif house_val == 'Rented':
        df['house_type_rented'] = 1
    
    return df

def prepare_features_for_prediction(df, for_regression=False):
    """
    Prepare features in the exact order expected by the model
    for_regression: if True, exclude max_monthly_emi (target variable for regression)
    """
    if for_regression:
        # For regression model - exclude max_monthly_emi (target variable)
        feature_order = [
            'age', 'gender', 'marital_status', 'monthly_salary',
            'years_of_employment', 'monthly_rent', 'family_size', 'dependents',
            'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities',
            'other_monthly_expenses', 'existing_loans', 'current_emi_amount',
            'credit_score', 'bank_balance', 'emergency_fund', 'emi_scenario',
            'requested_amount', 'requested_tenure',
            'debt_to_income_ratio', 'total_monthly_expenses', 'savings_capacity',
            'financial_stability', 'per_capita_income', 'employment_stability',
            'housing_burden_ratio', 'loan_to_income_ratio',
            'expance_to_income_ratio', 'affordability_ratio',
            'education_high school', 'education_post graduate',
            'education_professional', 'employment_type_private',
            'employment_type_self-employed', 'company_type_mid-size',
            'company_type_mnc', 'company_type_small', 'company_type_startup',
            'house_type_own', 'house_type_rented'
        ]
    else:
        # For classification model - include all features
        feature_order = [
            'age', 'gender', 'marital_status', 'monthly_salary',
            'years_of_employment', 'monthly_rent', 'family_size', 'dependents',
            'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities',
            'other_monthly_expenses', 'existing_loans', 'current_emi_amount',
            'credit_score', 'bank_balance', 'emergency_fund', 'emi_scenario',
            'requested_amount', 'requested_tenure', 'max_monthly_emi',
            'debt_to_income_ratio', 'total_monthly_expenses', 'savings_capacity',
            'financial_stability', 'per_capita_income', 'employment_stability',
            'housing_burden_ratio', 'loan_to_income_ratio',
            'expance_to_income_ratio', 'affordability_ratio',
            'education_high school', 'education_post graduate',
            'education_professional', 'employment_type_private',
            'employment_type_self-employed', 'company_type_mid-size',
            'company_type_mnc', 'company_type_small', 'company_type_startup',
            'house_type_own', 'house_type_rented'
        ]
    
    return df[feature_order]