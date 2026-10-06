"""Vercel Serverless Function entrypoint for RETENTIX AI Churn Prediction.

Exports:
  - `handler`: BaseHTTPRequestHandler subclass (for Vercel serverless functions)
  - `app` / `application`: Standard WSGI callable (for WSGI/ASGI runners)

Endpoints:
  - GET  /api or /api/health       -> Health check, active bundle info, sklearn version
  - GET  /api/model_info           -> Model architecture, feature count, decision threshold
  - GET  /api/stats                -> Executive KPIs & key segment churn insights
  - GET  /api/sample_csv           -> Sample CSV for batch inference testing
  - POST /api/predict              -> Single customer inference
  - POST /api/predict_batch        -> Multi-customer batch scoring from JSON or CSV
"""

import csv
import io
import json
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler

# Ensure project root is deterministically in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import sklearn

from backend.model import REQUIRED_COLS
from backend.prediction_service import (
    PredictionServiceError,
    get_prediction_service,
)

# Shared singleton prediction service instance
_SERVICE = None


def _get_service():
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = get_prediction_service()
    return _SERVICE


def _determine_risk_level(prob_pct: float) -> tuple[str, str, str]:
    """Return (label, color_hex, badge_class)."""
    if prob_pct < 30.0:
        return "Low Risk", "#10B981", "risk-low"
    elif prob_pct < 60.0:
        return "Medium Risk", "#F59E0B", "risk-medium"
    elif prob_pct < 80.0:
        return "High Risk", "#EF4444", "risk-high"
    else:
        return "Critical Risk", "#DC2626", "risk-critical"


def _generate_interventions(customer: dict, prob_pct: float) -> list[dict]:
    """Generate prioritized retention recommendations based on risk factors."""
    interventions = []
    contract = str(customer.get("Contract", ""))
    tech_support = str(customer.get("TechSupport", ""))
    internet = str(customer.get("InternetService", ""))
    payment = str(customer.get("PaymentMethod", ""))
    monthly = float(customer.get("MonthlyCharges", 0) or 0)

    if contract == "Month-to-month":
        interventions.append({
            "action": "Contract Upgrade Incentive",
            "impact": "Reduces churn risk by ~30%",
            "description": "Offer 15% discount on an annual contract lock-in.",
            "priority": "HIGH",
        })

    if internet == "Fiber optic" and tech_support == "No":
        interventions.append({
            "action": "Complimentary Premium Tech Support",
            "impact": "Reduces churn risk by ~18%",
            "description": "Include free 3-month VIP technical onboarding to resolve connectivity friction.",
            "priority": "HIGH",
        })

    if payment == "Electronic check":
        interventions.append({
            "action": "Autopay / Credit Card Migration",
            "impact": "Reduces payment churn by ~12%",
            "description": "Provide a $5 monthly bill credit for switching to automated bank transfer or credit card.",
            "priority": "MEDIUM",
        })

    if monthly > 80.0:
        interventions.append({
            "action": "Plan Right-Sizing Review",
            "impact": "Prevents bill-shock churn",
            "description": "Review unutilized streaming/add-on services and tailor a balanced loyalty bundle.",
            "priority": "MEDIUM",
        })

    if not interventions:
        interventions.append({
            "action": "Loyalty Appreciation",
            "impact": "Reinforces retention",
            "description": "Send quarterly satisfaction check-in and loyalty milestone reward.",
            "priority": "LOW",
        })

    return interventions


def dispatch_request(method: str, raw_path: str, headers: dict, body_bytes: bytes) -> tuple[int, dict, bytes]:
    """Core request dispatcher shared by both BaseHTTPRequestHandler and WSGI callable."""
    parsed = urllib.parse.urlparse(raw_path)
    path = parsed.path.rstrip("/")
    if not path:
        path = "/api"

    # Normalize paths like /api/index.py/predict -> /api/predict
    if path.startswith("/api/index.py"):
        path = "/api" + path[len("/api/index.py"):]

    res_headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
    }

    if method == "OPTIONS":
        return 200, res_headers, b""

    # --- ROUTE: GET /api or /api/health ---
    if method == "GET" and (path in ("/api", "/api/health", "/api/status")):
        try:
            svc = _get_service()
            meta = svc.metadata()
            payload = {
                "status": "healthy",
                "service": "RETENTIX AI Churn Prediction Engine",
                "active_bundle": svc.bundle_version,
                "model_name": svc.best_model_name,
                "model_type": meta["model_type"],
                "decision_threshold": meta["decision_threshold"],
                "features_count": meta["transformed_feature_count"],
                "scikit_learn_version": sklearn.__version__,
                "required_features": REQUIRED_COLS,
            }
            return 200, res_headers, json.dumps(payload, indent=2).encode("utf-8")
        except Exception as exc:
            err = {"status": "error", "error": str(exc)}
            return 500, res_headers, json.dumps(err).encode("utf-8")

    # --- ROUTE: GET /api/model_info ---
    if method == "GET" and path == "/api/model_info":
        try:
            svc = _get_service()
            meta = svc.metadata()
            return 200, res_headers, json.dumps({"status": "ok", "metadata": meta}).encode("utf-8")
        except Exception as exc:
            return 500, res_headers, json.dumps({"status": "error", "message": str(exc)}).encode("utf-8")

    # --- ROUTE: GET /api/stats ---
    if method == "GET" and path in ("/api/stats", "/api/dataset_stats"):
        stats = {
            "total_customers": 7043,
            "overall_churn_rate_pct": 26.54,
            "monthly_revenue_at_risk": 139130.85,
            "annual_revenue_at_risk": 1669570.20,
            "key_vulnerabilities": [
                {
                    "segment": "Month-to-Month Contracts",
                    "churn_rate_pct": 42.7,
                    "insight": "Converting 15% to 1-Year plans recovers ~$463k/yr.",
                    "severity": "CRITICAL",
                },
                {
                    "segment": "Fiber Optic Users without Tech Support",
                    "churn_rate_pct": 41.9,
                    "insight": "Tech support gap is the primary driver of fiber churn.",
                    "severity": "HIGH",
                },
                {
                    "segment": "Electronic Check Payment",
                    "churn_rate_pct": 45.3,
                    "insight": "Highest friction payment channel; autopay migration recommended.",
                    "severity": "HIGH",
                },
            ],
        }
        return 200, res_headers, json.dumps(stats, indent=2).encode("utf-8")

    # --- ROUTE: GET /api/sample_csv ---
    if method == "GET" and path == "/api/sample_csv":
        csv_sample = (
            "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,"
            "InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,"
            "StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n"
            "7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85\n"
            "5575-GNVDE,Male,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.5\n"
            "3668-QPYBK,Male,0,No,No,2,Yes,No,DSL,Yes,Yes,No,No,No,No,Month-to-month,Yes,Mailed check,53.85,108.15\n"
            "7795-CFOCW,Male,0,No,No,45,No,No phone service,DSL,Yes,No,Yes,Yes,No,No,One year,No,Bank transfer (automatic),42.30,1840.75\n"
            "9237-HQITU,Female,0,No,No,2,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,70.70,151.65\n"
            "9305-CDSKC,Female,0,No,No,8,Yes,Yes,Fiber optic,No,No,Yes,No,Yes,Yes,Month-to-month,Yes,Electronic check,99.65,820.5\n"
            "1452-KIOVK,Male,0,No,Yes,22,Yes,Yes,Fiber optic,No,Yes,No,No,Yes,No,Month-to-month,Yes,Credit card (automatic),89.10,1949.4\n"
            "6713-OKOMC,Female,0,No,No,10,No,No phone service,DSL,Yes,No,No,No,No,No,Month-to-month,No,Mailed check,29.75,301.9\n"
        )
        res_headers["Content-Type"] = "text/csv"
        res_headers["Content-Disposition"] = 'attachment; filename="sample_churn_data.csv"'
        return 200, res_headers, csv_sample.encode("utf-8")

    # --- ROUTE: POST /api/predict (Single Inference) ---
    if method == "POST" and path == "/api/predict":
        try:
            body_str = body_bytes.decode("utf-8") if body_bytes else "{}"
            req_data = json.loads(body_str) if body_str else {}
        except Exception:
            return 400, res_headers, json.dumps({"error": "Invalid JSON payload"}).encode("utf-8")

        try:
            svc = _get_service()
            # Predict single
            res = svc.predict_single(req_data)
            prob = res["churn_probability"]
            pred = res["prediction"]
            risk_label, color, badge_cls = _determine_risk_level(prob)
            interventions = _generate_interventions(req_data, prob)

            resp_payload = {
                "success": True,
                "prediction": pred,
                "churn": pred == "Yes",
                "churn_probability": prob,
                "no_churn_probability": res["no_churn_probability"],
                "risk_level": risk_label,
                "risk_color": color,
                "badge_class": badge_cls,
                "active_bundle": svc.bundle_version,
                "model_name": svc.best_model_name,
                "interventions": interventions,
            }
            return 200, res_headers, json.dumps(resp_payload).encode("utf-8")
        except PredictionServiceError as exc:
            return 422, res_headers, json.dumps({"error": "Contract/Prediction error", "details": str(exc)}).encode("utf-8")
        except Exception as exc:
            return 500, res_headers, json.dumps({"error": "Internal error during prediction", "details": str(exc)}).encode("utf-8")

    # --- ROUTE: POST /api/predict_batch (Batch Inference) ---
    if method == "POST" and (path in ("/api/predict_batch", "/api/batch")):
        try:
            svc = _get_service()
            body_str = body_bytes.decode("utf-8", errors="replace") if body_bytes else ""
            content_type = headers.get("content-type", headers.get("Content-Type", ""))

            df = None
            if "text/csv" in content_type or body_str.strip().startswith("customerID,") or "tenure," in body_str:
                df = pd.read_csv(io.StringIO(body_str))
            else:
                data = json.loads(body_str) if body_str else {}
                if "csv_content" in data:
                    df = pd.read_csv(io.StringIO(data["csv_content"]))
                elif "records" in data and isinstance(data["records"], list):
                    df = pd.DataFrame(data["records"])
                elif isinstance(data, list):
                    df = pd.DataFrame(data)

            if df is None or len(df) == 0:
                return 400, res_headers, json.dumps({"error": "No records or CSV content provided"}).encode("utf-8")

            # Score using batch inference service
            labels, probs = svc.predict_batch(df)

            customer_ids = df["customerID"].tolist() if "customerID" in df.columns else [f"CUST-{i+1:04d}" for i in range(len(df))]
            results = []
            churn_count = 0
            critical_high_count = 0

            for idx, (cid, lbl, p) in enumerate(zip(customer_ids, labels, probs)):
                p_pct = round(float(p) * 100, 2)
                is_churn = str(lbl) == "Yes"
                if is_churn:
                    churn_count += 1
                risk_label, color, _ = _determine_risk_level(p_pct)
                if p_pct >= 60.0:
                    critical_high_count += 1

                record_summary = {
                    "row": idx + 1,
                    "customerID": str(cid),
                    "prediction": str(lbl),
                    "churn": is_churn,
                    "churn_probability": p_pct,
                    "risk_level": risk_label,
                    "risk_color": color,
                }
                # Include key contextual attributes if present
                for col in ["tenure", "Contract", "MonthlyCharges", "InternetService"]:
                    if col in df.columns:
                        val = df.iloc[idx][col]
                        record_summary[col] = float(val) if isinstance(val, (int, float, np.number)) else str(val)

                results.append(record_summary)

            total = len(df)
            churn_pct = round((churn_count / total) * 100, 2) if total > 0 else 0.0

            batch_resp = {
                "success": True,
                "total_records": total,
                "churn_count": churn_count,
                "retain_count": total - churn_count,
                "churn_rate_pct": churn_pct,
                "critical_high_risk_count": critical_high_count,
                "active_bundle": svc.bundle_version,
                "records": results,
            }
            return 200, res_headers, json.dumps(batch_resp).encode("utf-8")
        except Exception as exc:
            return 500, res_headers, json.dumps({"error": "Batch scoring failed", "details": str(exc)}).encode("utf-8")

    # Default 404
    return 404, res_headers, json.dumps({"error": "Endpoint not found", "path": path}).encode("utf-8")


class handler(BaseHTTPRequestHandler):
    """Vercel Serverless Function entrypoint (BaseHTTPRequestHandler)."""

    def do_OPTIONS(self):
        status, headers, body = dispatch_request("OPTIONS", self.path, dict(self.headers), b"")
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        status, headers, body = dispatch_request("GET", self.path, dict(self.headers), b"")
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body_in = self.rfile.read(content_length) if content_length > 0 else b""
        status, headers, body = dispatch_request("POST", self.path, dict(self.headers), body_in)
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)


def app(environ, start_response):
    """Standard WSGI entrypoint callable for Vercel/Gunicorn/WSGI runtimes."""
    method = environ.get("REQUEST_METHOD", "GET")
    path = environ.get("PATH_INFO", "/")
    if environ.get("QUERY_STRING"):
        path += "?" + environ["QUERY_STRING"]

    try:
        content_length = int(environ.get("CONTENT_LENGTH") or 0)
    except (ValueError, TypeError):
        content_length = 0

    body_in = b""
    if content_length > 0 and "wsgi.input" in environ:
        body_in = environ["wsgi.input"].read(content_length)

    status_code, headers, body_out = dispatch_request(method, path, environ, body_in)
    status_str = f"{status_code} OK" if status_code == 200 else f"{status_code} Response"
    start_response(status_str, list(headers.items()))
    return [body_out]


# Alias application to app
application = app


if __name__ == "__main__":
    from http.server import HTTPServer

    port = int(os.environ.get("PORT", 8000))
    print(f"⚡ RETENTIX AI Vercel API Server running locally on http://localhost:{port}")
    httpd = HTTPServer(("0.0.0.0", port), handler)
    httpd.serve_forever()
