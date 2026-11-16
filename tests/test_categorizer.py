"""
Unit and Integration Tests for News Category & Topic Classification Engine
========================================================================
Validates that news text is accurately classified across the 7 enterprise taxonomies:
  - Politics
  - Technology
  - Business
  - Sports
  - Science
  - Entertainment
  - International News
"""

import pytest
from fastapi.testclient import TestClient
from News_Detection.categorizer import (
    CATEGORIES,
    predict_news_category,
    normalize_text,
)
from News_Detection.health_api import app
from News_Detection.ensemble import analyze_text
from News_Detection.batch_processor import BatchJobManager


client = TestClient(app)

REQUIRED_CATEGORIES = [
    "Politics",
    "Technology",
    "Business",
    "Sports",
    "Science",
    "Entertainment",
    "International News",
]


def test_required_categories_exact_match():
    """Verifies that all 7 required taxonomies are defined in the engine."""
    assert set(CATEGORIES) == set(REQUIRED_CATEGORIES)
    assert len(CATEGORIES) == 7


@pytest.mark.parametrize(
    "headline, expected_category",
    [
        (
            "Senate passes bipartisan legislation following heated debate in Congress with prime minister.",
            "Politics",
        ),
        (
            "Semiconductor chipmaker Nvidia unveils next-generation AI GPU microchip hardware and deep learning algorithm.",
            "Technology",
        ),
        (
            "Federal Reserve evaluates interest rates and inflation as Wall Street stocks rally on quarterly earnings report.",
            "Business",
        ),
        (
            "Real Madrid clinches Champions League trophy after penalty shootout in thrilling football tournament championship.",
            "Sports",
        ),
        (
            "NASA James Webb Space Telescope discovers exoplanet atmosphere with cosmic physics and telescope telemetry.",
            "Science",
        ),
        (
            "Hollywood director receives Academy Award Oscar nomination for blockbuster film premiere and soundtrack.",
            "Entertainment",
        ),
        (
            "United Nations Security Council negotiates bilateral ceasefire treaty and diplomatic accords in Geneva summit.",
            "International News",
        ),
    ],
)
def test_predict_news_category_all_domains(headline, expected_category):
    """Verifies accurate classification for distinct representative news headlines."""
    res = predict_news_category(headline)
    assert res["category"] == expected_category
    assert res["confidence"] > 25.0
    assert "distribution" in res
    assert len(res["distribution"]) == 7
    # Distribution sums to 100%
    assert 99.0 <= sum(res["distribution"].values()) <= 101.0
    assert "runner_up" in res
    assert "matched_signals" in res
    assert len(res["matched_signals"]) > 0


def test_predict_news_category_multilingual():
    """Verifies category classification handles multilingual terms correctly."""
    # Hindi international diplomacy
    hindi_text = "अंतरराष्ट्रीय कूटनीति और संयुक्त राष्ट्र में शांति समझौता हुआ।"
    res_hindi = predict_news_category(hindi_text)
    assert res_hindi["category"] == "International News"

    # Spanish political legislation
    spanish_text = "El presidente anuncia nueva legislación política en el congreso y parlamento."
    res_spanish = predict_news_category(spanish_text)
    assert res_spanish["category"] == "Politics"

    # French scientific research
    french_text = "La recherche scientifique au laboratoire découvre de nouvelles particules de physique quantique."
    res_french = predict_news_category(french_text)
    assert res_french["category"] == "Science"


def test_predict_news_category_empty_fallback():
    """Verifies that empty or whitespace input returns graceful neutral distribution."""
    res = predict_news_category("")
    assert res["category"] in CATEGORIES
    assert res["is_ambiguous"] is True
    assert res["matched_signals"] == []
    for cat in CATEGORIES:
        assert cat in res["distribution"]


def test_predict_news_category_temperature_scaling():
    """Verifies that temperature scaling smooths the probability distribution."""
    text = "Artificial intelligence deep learning software algorithm microchip GPU."
    high_temp = predict_news_category(text, temperature=3.0)
    low_temp = predict_news_category(text, temperature=0.5)

    assert high_temp["category"] == "Technology"
    assert low_temp["category"] == "Technology"
    # Low temperature should produce higher peak probability than high temperature
    assert low_temp["confidence"] > high_temp["confidence"]


def test_api_category_endpoint_success():
    """Verifies POST /api/category FastAPI endpoint returns valid JSON."""
    response = client.post(
        "/api/category",
        json={"text": "Apple introduces breakthrough AI microchips and neural computing algorithms."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Technology"
    assert "confidence" in data
    assert "distribution" in data
    assert len(data["distribution"]) == 7


def test_api_category_endpoint_empty_bad_request():
    """Verifies POST /api/category returns 422 or 400 for empty/whitespace payload."""
    response = client.post("/api/category", json={"text": "   "})
    assert response.status_code in (400, 422)


def test_ensemble_analyze_text_includes_category():
    """Verifies that analyze_text enriches the payload with category metadata."""
    from unittest.mock import MagicMock
    mock_p1 = MagicMock(return_value=[{"label": "REAL", "score": 0.90}])
    mock_p2 = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.85}])
    mock_p3 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.88}])
    dummy_pipes = (mock_p1, mock_p2, mock_p3)
    result = analyze_text(
        "Quarterly corporate earnings and stock market shares on Wall Street rally amid Federal Reserve interest rates.",
        pipes=dummy_pipes,
        enable_grounding=False,
    )
    assert "category" in result
    cat_pred = result["category"]
    assert cat_pred["category"] == "Business"
    assert "confidence" in cat_pred
    assert "distribution" in cat_pred


def test_batch_processor_records_category():
    """Verifies that batch processor persists category and summary counts."""
    from News_Detection.storage import BatchJobStore, SQLiteManager
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        tmp_db_path = tmp_db.name

    try:
        db_mgr = SQLiteManager(db_path=tmp_db_path)
        custom_store = BatchJobStore(db_manager=db_mgr)
        manager = BatchJobManager(job_store=custom_store)

        items = [
            {"id": "doc-1", "text": "Congress passes budget legislation in political vote.", "metadata": {}},
            {"id": "doc-2", "text": "NASA satellite and telescope explore exoplanet physics.", "metadata": {}},
        ]
        job_id = manager.create_job(
            filename="test_news.csv",
            items=items,
            metadata={"total": 2},
            enable_grounding=False,
        )

        manager.run_job_sync(job_id)
        job = manager.get_job(job_id)

        assert job["status"] == "completed"
        assert len(job["results"]) == 2
        categories = [r.get("category") for r in job["results"]]
        assert "Politics" in categories
        assert "Science" in categories

        # Summary check
        summary = job.get("summary", {})
        assert "categories" in summary
        assert summary["categories"].get("Politics", 0) >= 1
        assert summary["categories"].get("Science", 0) >= 1

        # CSV export check
        csv_out = manager.export_csv(job_id)
        assert "category" in csv_out
        assert "Politics" in csv_out or "Science" in csv_out
    finally:
        if os.path.exists(tmp_db_path):
            os.remove(tmp_db_path)
