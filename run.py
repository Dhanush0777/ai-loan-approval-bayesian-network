"""
Launcher script for AI-Based Loan Approval Prediction Using Bayesian Network.
"""

import sys
import os
import webbrowser

# Add backend directory to sys.path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    host = '127.0.0.1'
    url = f"http://{host}:{port}"
    
    print("=" * 70)
    print("AI-BASED LOAN APPROVAL PREDICTION USING BAYESIAN NETWORK")
    print("Institutional Bayesian Risk Underwriting & Explainable AI System")
    print("=" * 70)
    print(f"[*] Starting local server on: {url}")
    print("[*] Press Ctrl+C to terminate.")
    print("=" * 70)

    # Optional auto-open browser
    if os.environ.get('AUTO_OPEN_BROWSER', 'true').lower() == 'true':
        try:
            webbrowser.open(url)
        except Exception:
            pass

    app.run(host=host, port=port, debug=False)
