"""
Database layer for AI-Based Loan Approval Prediction system.
Uses SQLite for persistent storage of loan evaluations and analytics.
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
import random

DB_PATH = os.path.join(os.path.dirname(__file__), 'loan_predictions.db')


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            applicant_name TEXT NOT NULL,
            age INTEGER,
            monthly_income REAL NOT NULL,
            employment_status TEXT NOT NULL,
            employment_duration REAL NOT NULL,
            credit_score INTEGER NOT NULL,
            existing_loans TEXT NOT NULL,
            existing_monthly_debt REAL NOT NULL,
            loan_amount REAL NOT NULL,
            loan_tenure INTEGER,
            repayment_history TEXT NOT NULL,
            debt_to_income_ratio REAL NOT NULL,
            approval_probability REAL NOT NULL,
            rejection_probability REAL NOT NULL,
            risk_level TEXT NOT NULL,
            prediction TEXT NOT NULL,
            decision_code TEXT NOT NULL,
            key_positive_factors TEXT,
            key_warning_factors TEXT
        )
    ''')
    conn.commit()

    # Check if empty; if so, pre-seed sample history for dashboard
    cursor.execute('SELECT COUNT(*) as cnt FROM predictions')
    count = cursor.fetchone()['cnt']
    if count == 0:
        seed_sample_data(conn)

    conn.close()


def seed_sample_data(conn):
    cursor = conn.cursor()
    now = datetime.now()

    sample_records = [
        ("Sarah Jenkins", 34, 85000, "Salaried", 6.0, 790, "No", 8000, 450000, 36, "Excellent", 9.4, 91.2, 8.8, "LOW", "HIGH PROBABILITY OF APPROVAL", "APPROVED",
         ["Excellent credit score", "Long employment stability", "Low DTI ratio"], ["None"]),
        ("Marcus Vance", 42, 28000, "Self-employed", 1.2, 540, "Yes", 15000, 950000, 60, "Poor", 53.6, 7.8, 92.2, "HIGH", "HIGH PROBABILITY OF REJECTION", "REJECTED",
         ["None"], ["Poor credit score", "High DTI ratio", "Past default history"]),
        ("Elena Rostova", 29, 62000, "Salaried", 4.0, 710, "Yes", 14000, 380000, 48, "Good", 22.5, 68.4, 31.6, "MEDIUM", "MODERATE / CONDITIONAL APPROVAL", "REVIEW",
         ["Good credit score", "Salaried employment"], ["Active loan obligations"]),
        ("David Chen", 45, 120000, "Salaried", 10.0, 820, "No", 12000, 600000, 60, "Excellent", 10.0, 95.8, 4.2, "LOW", "HIGH PROBABILITY OF APPROVAL", "APPROVED",
         ["Exceptional credit score", "Executive income level", "Flawless repayment"], ["None"]),
        ("Aisha Patel", 26, 32000, "Unemployed", 0.5, 590, "Yes", 16000, 500000, 36, "Average", 50.0, 11.4, 88.6, "HIGH", "HIGH PROBABILITY OF REJECTION", "REJECTED",
         ["None"], ["Currently unemployed", "High debt-to-income", "Existing debts"]),
        ("James Wilson", 38, 75000, "Salaried", 5.5, 745, "No", 9500, 520000, 48, "Good", 12.7, 85.2, 14.8, "LOW", "HIGH PROBABILITY OF APPROVAL", "APPROVED",
         ["High credit score", "Solid employment tenure"], ["None"]),
        ("Priya Sharma", 31, 48000, "Self-employed", 3.0, 660, "Yes", 13000, 350000, 36, "Good", 27.1, 56.2, 43.8, "MEDIUM", "MODERATE / CONDITIONAL APPROVAL", "REVIEW",
         ["Moderate DTI ratio", "Good repayment track"], ["Fair credit tier"]),
        ("Carlos Mendez", 50, 41000, "Salaried", 8.0, 610, "Yes", 19000, 800000, 60, "Average", 46.3, 24.5, 75.5, "HIGH", "HIGH PROBABILITY OF REJECTION", "REJECTED",
         ["Stable employment tenure"], ["High loan amount", "High DTI", "Fair credit"]),
        ("Emily Watson", 33, 92000, "Salaried", 7.0, 805, "No", 7000, 400000, 24, "Excellent", 7.6, 94.1, 5.9, "LOW", "HIGH PROBABILITY OF APPROVAL", "APPROVED",
         ["Top-tier credit score", "Low debt burden", "High income"], ["None"]),
        ("Ahmed Al-Mansoor", 36, 58000, "Salaried", 3.5, 680, "No", 11000, 320000, 36, "Good", 18.9, 72.8, 27.2, "LOW", "HIGH PROBABILITY OF APPROVAL", "APPROVED",
         ["Clean liability history", "Good credit score"], ["Moderate tenure"]),
        ("Robert Taylor", 40, 35000, "Self-employed", 2.0, 560, "Yes", 17500, 650000, 48, "Poor", 50.0, 10.3, 89.7, "HIGH", "HIGH PROBABILITY OF REJECTION", "REJECTED",
         ["None"], ["Poor credit score", "Repeated missed payments", "High DTI"]),
        ("Sophia Lin", 27, 65000, "Salaried", 3.0, 725, "No", 8500, 280000, 24, "Good", 13.1, 84.7, 15.3, "LOW", "HIGH PROBABILITY OF APPROVAL", "APPROVED",
         ["Modest loan request", "Solid credit score", "Salaried status"], ["None"])
    ]

    for i, r in enumerate(sample_records):
        record_time = (now - timedelta(days=random.randint(1, 20), hours=random.randint(1, 23))).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO predictions (
                timestamp, applicant_name, age, monthly_income, employment_status,
                employment_duration, credit_score, existing_loans, existing_monthly_debt,
                loan_amount, loan_tenure, repayment_history, debt_to_income_ratio,
                approval_probability, rejection_probability, risk_level, prediction,
                decision_code, key_positive_factors, key_warning_factors
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record_time, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9],
            r[10], r[11], r[12], r[13], r[14], r[15], r[16], json.dumps(r[17]), json.dumps(r[18])
        ))

    conn.commit()


def save_prediction(record_data):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO predictions (
            timestamp, applicant_name, age, monthly_income, employment_status,
            employment_duration, credit_score, existing_loans, existing_monthly_debt,
            loan_amount, loan_tenure, repayment_history, debt_to_income_ratio,
            approval_probability, rejection_probability, risk_level, prediction,
            decision_code, key_positive_factors, key_warning_factors
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp,
        record_data.get('applicant_name', 'Applicant'),
        int(record_data.get('age', 30)),
        float(record_data.get('monthly_income', 0)),
        record_data.get('employment_status', 'Salaried'),
        float(record_data.get('employment_duration', 0)),
        int(record_data.get('credit_score', 650)),
        record_data.get('existing_loans', 'No'),
        float(record_data.get('existing_monthly_debt', 0)),
        float(record_data.get('loan_amount', 0)),
        int(record_data.get('loan_tenure', 36)),
        record_data.get('repayment_history', 'Good'),
        float(record_data.get('computed_dti', 0)),
        float(record_data.get('approval_probability', 0)),
        float(record_data.get('rejection_probability', 0)),
        record_data.get('risk_level', 'MEDIUM'),
        record_data.get('prediction', ''),
        record_data.get('decision_code', 'REVIEW'),
        json.dumps(record_data.get('key_positive_factors', [])),
        json.dumps(record_data.get('key_warning_factors', []))
    ))
    
    inserted_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return inserted_id


def get_all_predictions(limit=50, search=""):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if search:
        query = '''
            SELECT * FROM predictions 
            WHERE applicant_name LIKE ? OR risk_level LIKE ? OR prediction LIKE ?
            ORDER BY id DESC LIMIT ?
        '''
        cursor.execute(query, (f'%{search}%', f'%{search}%', f'%{search}%', limit))
    else:
        cursor.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT ?', (limit,))
        
    rows = cursor.fetchall()
    predictions = [dict(row) for row in rows]
    
    for p in predictions:
        try:
            p['key_positive_factors'] = json.loads(p['key_positive_factors']) if p['key_positive_factors'] else []
        except:
            p['key_positive_factors'] = []
        try:
            p['key_warning_factors'] = json.loads(p['key_warning_factors']) if p['key_warning_factors'] else []
        except:
            p['key_warning_factors'] = []
            
    conn.close()
    return predictions


def delete_prediction(prediction_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions WHERE id = ?', (prediction_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def get_dashboard_statistics():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) as total FROM predictions')
    total = cursor.fetchone()['total']
    
    if total == 0:
        conn.close()
        return {
            'total_applications': 0,
            'approved_count': 0,
            'review_count': 0,
            'rejected_count': 0,
            'avg_approval_prob': 0,
            'high_risk_count': 0,
            'risk_distribution': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0},
            'recent_predictions': []
        }

    cursor.execute("SELECT COUNT(*) as cnt FROM predictions WHERE decision_code = 'APPROVED'")
    approved = cursor.fetchone()['cnt']

    cursor.execute("SELECT COUNT(*) as cnt FROM predictions WHERE decision_code = 'REVIEW'")
    review = cursor.fetchone()['cnt']

    cursor.execute("SELECT COUNT(*) as cnt FROM predictions WHERE decision_code = 'REJECTED'")
    rejected = cursor.fetchone()['cnt']

    cursor.execute('SELECT AVG(approval_probability) as avg_prob FROM predictions')
    avg_prob = cursor.fetchone()['avg_prob'] or 0

    cursor.execute("SELECT COUNT(*) as cnt FROM predictions WHERE risk_level = 'HIGH'")
    high_risk = cursor.fetchone()['cnt']

    cursor.execute("SELECT COUNT(*) as cnt FROM predictions WHERE risk_level = 'MEDIUM'")
    medium_risk = cursor.fetchone()['cnt']

    cursor.execute("SELECT COUNT(*) as cnt FROM predictions WHERE risk_level = 'LOW'")
    low_risk = cursor.fetchone()['cnt']

    # Credit score tiers vs approval rate
    cursor.execute('''
        SELECT 
            CASE 
                WHEN credit_score < 580 THEN 'Poor (<580)'
                WHEN credit_score < 670 THEN 'Fair (580-669)'
                WHEN credit_score < 740 THEN 'Good (670-739)'
                ELSE 'Excellent (740+)'
            END as tier,
            COUNT(*) as count,
            AVG(approval_probability) as avg_prob
        FROM predictions
        GROUP BY tier
    ''')
    credit_tiers = [dict(r) for r in cursor.fetchall()]

    # Income brackets vs approval rate
    cursor.execute('''
        SELECT 
            CASE 
                WHEN monthly_income < 35000 THEN 'Low (<35k)'
                WHEN monthly_income <= 80000 THEN 'Medium (35k-80k)'
                ELSE 'High (>80k)'
            END as bracket,
            COUNT(*) as count,
            AVG(approval_probability) as avg_prob
        FROM predictions
        GROUP BY bracket
    ''')
    income_brackets = [dict(r) for r in cursor.fetchall()]

    # Recent 5 predictions
    cursor.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT 5')
    recent = [dict(r) for r in cursor.fetchall()]

    conn.close()

    return {
        'total_applications': total,
        'approved_count': approved,
        'review_count': review,
        'rejected_count': rejected,
        'avg_approval_prob': round(avg_prob, 1),
        'high_risk_count': high_risk,
        'risk_distribution': {
            'LOW': low_risk,
            'MEDIUM': medium_risk,
            'HIGH': high_risk
        },
        'credit_tiers': credit_tiers,
        'income_brackets': income_brackets,
        'recent_predictions': recent
    }
