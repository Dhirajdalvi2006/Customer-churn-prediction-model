"""Tests for Vercel Python API entrypoint (api/index.py)."""

import json
import pytest
from api.index import dispatch_request, handler, app, application


def test_api_health_endpoint():
    status, headers, body = dispatch_request("GET", "/api/health", {}, b"")
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    assert data["status"] == "healthy"
    assert data["active_bundle"] == "v2"
    assert "scikit_learn_version" in data


def test_api_model_info_endpoint():
    status, headers, body = dispatch_request("GET", "/api/model_info", {}, b"")
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    assert data["status"] == "ok"
    assert "metadata" in data


def test_api_stats_endpoint():
    status, headers, body = dispatch_request("GET", "/api/stats", {}, b"")
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    assert data["total_customers"] == 7043
    assert len(data["key_vulnerabilities"]) > 0


def test_api_predict_single():
    sample = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 92.5,
        "TotalCharges": 185.0,
    }
    status, headers, body = dispatch_request(
        "POST", "/api/predict", {}, json.dumps(sample).encode("utf-8")
    )
    assert status == 200
    res = json.loads(body.decode("utf-8"))
    assert res["success"] is True
    assert res["prediction"] in ("Yes", "No")
    assert 0.0 <= res["churn_probability"] <= 100.0
    assert "interventions" in res


def test_api_predict_batch_csv():
    batch_csv = (
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,"
        "InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,"
        "StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n"
        "CUST-001,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85\n"
        "CUST-002,Male,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.5\n"
    )
    status, headers, body = dispatch_request(
        "POST", "/api/predict_batch", {"content-type": "text/csv"}, batch_csv.encode("utf-8")
    )
    assert status == 200
    res = json.loads(body.decode("utf-8"))
    assert res["success"] is True
    assert res["total_records"] == 2
    assert len(res["records"]) == 2


def test_api_wsgi_callable():
    environ = {
        "REQUEST_METHOD": "GET",
        "PATH_INFO": "/api/health",
        "wsgi.input": None,
    }
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = headers

    body_list = app(environ, start_response)
    assert captured["status"] == "200 OK"
    data = json.loads(body_list[0].decode("utf-8"))
    assert data["status"] == "healthy"
