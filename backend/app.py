"""
Flask Application and REST API for AI-Based Loan Approval Prediction Using Bayesian Network.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys

# Ensure backend folder is on path
sys.path.insert(0, os.path.dirname(__file__))

from bayesian_model import LoanBayesianModel
import database

# Initialize Flask app
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app = Flask(__name__, static_folder=frontend_dir, static_url_path='')
CORS(app)

# Initialize Bayesian Model and Database
model = LoanBayesianModel()
database.init_db()


# Sample applicants specified in prompt
SAMPLE_APPLICANTS = {
    'sample_1': {
        'name': 'Sample Applicant 1 (Prime / High Approval)',
        'applicant_name': 'Eleanor Vance',
        'age': 35,
        'monthly_income': 70000,
        'credit_score': 780,
        'employment_status': 'Salaried',
        'existing_loans': 'No',
        'existing_monthly_debt': 7000,
        'loan_amount': 500000,
        'loan_tenure': 48,
        'repayment_history': 'Excellent',
        'employment_duration': 5,
        'notes': 'High income, excellent credit score (780), 5 years salaried, clean debt history.'
    },
    'sample_2': {
        'name': 'Sample Applicant 2 (Subprime / High Risk)',
        'applicant_name': 'Travis Miller',
        'age': 28,
        'monthly_income': 30000,
        'credit_score': 580,
        'employment_status': 'Self-employed',
        'existing_loans': 'Yes',
        'existing_monthly_debt': 16000,
        'loan_amount': 1000000,
        'loan_tenure': 60,
        'repayment_history': 'Poor',
        'employment_duration': 1,
        'notes': 'Low income, poor credit (580), high requested loan amount, high DTI (53%), previous defaults.'
    },
    'sample_3': {
        'name': 'Sample Applicant 3 (Moderate / Borderline)',
        'applicant_name': 'Kavita Menon',
        'age': 32,
        'monthly_income': 50000,
        'credit_score': 700,
        'employment_status': 'Salaried',
        'existing_loans': 'Yes',
        'existing_monthly_debt': 14000,
        'loan_amount': 400000,
        'loan_tenure': 36,
        'repayment_history': 'Good',
        'employment_duration': 3,
        'notes': 'Moderate income, fair-to-good credit (700), salaried, moderate loan request.'
    }
}


@app.route('/')
def serve_index():
    return send_from_directory(frontend_dir, 'index.html')


@app.route('/api/predict', methods=['POST'])
def run_prediction():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No input payload provided'}), 400

        # Validate input ranges
        credit_score = float(data.get('credit_score', 650))
        if credit_score < 300 or credit_score > 850:
            return jsonify({'error': 'Credit score must be between 300 and 850'}), 400

        monthly_income = float(data.get('monthly_income', 0))
        if monthly_income <= 0:
            return jsonify({'error': 'Monthly income must be greater than 0'}), 400

        loan_amount = float(data.get('loan_amount', 0))
        if loan_amount <= 0:
            return jsonify({'error': 'Loan amount must be greater than 0'}), 400

        # Run exact Bayesian Inference
        result = model.predict(data)

        # Extract positive and warning factors for database
        pos_factors = [f['text'] for f in result['explanations'] if f['type'] == 'positive']
        warn_factors = [f['text'] for f in result['explanations'] if f['type'] == 'warning']

        record_payload = {
            'applicant_name': data.get('applicant_name', 'Anonymous Applicant'),
            'age': data.get('age', 30),
            'monthly_income': monthly_income,
            'employment_status': data.get('employment_status', 'Salaried'),
            'employment_duration': data.get('employment_duration', 2),
            'credit_score': credit_score,
            'existing_loans': data.get('existing_loans', 'No'),
            'existing_monthly_debt': data.get('existing_monthly_debt', 0),
            'loan_amount': loan_amount,
            'loan_tenure': data.get('loan_tenure', 36),
            'repayment_history': data.get('repayment_history', 'Good'),
            'computed_dti': result['computed_dti'],
            'approval_probability': result['approval_probability'],
            'rejection_probability': result['rejection_probability'],
            'risk_level': result['risk_level'],
            'prediction': result['prediction'],
            'decision_code': result['decision_code'],
            'key_positive_factors': pos_factors,
            'key_warning_factors': warn_factors
        }

        # Persist to SQLite
        record_id = database.save_prediction(record_payload)
        result['record_id'] = record_id
        result['applicant_data'] = record_payload

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/what-if', methods=['POST'])
def run_what_if():
    """
    Evaluates dynamic counterfactual prediction without persisting to database.
    Employs Jeffrey's soft evidence conditioning for continuous variables to ensure
    smooth, immediate feedback on slider movement with exact Bayesian inference.
    """
    try:
        data = request.get_json(force=True)
        original_data = data.get('original')
        modified_data = data.get('modified')

        if not original_data or not modified_data:
            return jsonify({'error': 'Both original and modified data payloads are required'}), 400

        result = model.predict_what_if(original_data, modified_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """
    Returns the network nodes, edges, states, and graph topology.
    """
    try:
        schema = model.get_network_schema()
        return jsonify(schema)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cpt/<node_name>', methods=['GET'])
def get_cpt(node_name):
    """
    Returns the exact Conditional Probability Table for a requested node.
    """
    try:
        cpt = model.get_cpt(node_name)
        if not cpt:
            return jsonify({'error': f'Node {node_name} not found in Bayesian Network'}), 404
        return jsonify(cpt)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/samples', methods=['GET'])
def get_samples():
    """
    Returns pre-configured sample applicants.
    """
    return jsonify(SAMPLE_APPLICANTS)


@app.route('/api/dashboard-stats', methods=['GET'])
def get_stats():
    """
    Returns summary statistics for the dashboard charts.
    """
    try:
        stats = database.get_dashboard_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """
    Returns list of saved predictions.
    """
    try:
        search = request.args.get('search', '')
        limit = int(request.args.get('limit', 50))
        records = database.get_all_predictions(limit=limit, search=search)
        return jsonify({'predictions': records, 'total': len(records)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history/<int:record_id>', methods=['DELETE'])
def delete_record(record_id):
    """
    Deletes an individual prediction record.
    """
    try:
        success = database.delete_prediction(record_id)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset-demo', methods=['POST'])
def reset_demo():
    """
    Resets database with fresh sample historical data.
    """
    try:
        conn = database.get_db_connection()
        conn.execute('DELETE FROM predictions')
        conn.commit()
        database.seed_sample_data(conn)
        conn.close()
        return jsonify({'success': True, 'message': 'Demo database successfully reset'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f"Starting AI Bayesian Network Loan Approval Server on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
