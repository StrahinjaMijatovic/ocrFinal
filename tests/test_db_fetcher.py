import json
import pytest
from unittest.mock import MagicMock, patch
from app.pipeline.db_fetcher import fetch_request_data


SAMPLE_DB_JSON = {
    "zahtev_id": 438281,
    "zahtev_kod": "VR-2024-001",
    "datum_podnosenja": "2024-03-01",
    "je_maloletnik": 0,
    "urgency_kategorija": "URGENCY_HIGH",
    "starosna_kategorija": "APPLICANT_STANDARD",
    "pasos_status": "PASSPORT_OK",
    "duzina_boravka_dana": 6,
    "dana_do_dolaska": 5,
    "pasos_istice_za_dana": 400,
    "broj_prethodnih_zahteva": 0,
    "starost": 30,
    "ime": "Test",
    "prezime": "User",
}


def test_fetch_returns_dict(mocker):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = (json.dumps(SAMPLE_DB_JSON),)
    mocker.patch("app.pipeline.db_fetcher.get_db_connection", return_value=mock_conn)

    result = fetch_request_data(438281)
    assert result["zahtev_id"] == 438281
    assert result["urgency_kategorija"] == "URGENCY_HIGH"


def test_fetch_raises_on_missing(mocker):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = (None,)
    mocker.patch("app.pipeline.db_fetcher.get_db_connection", return_value=mock_conn)

    with pytest.raises(ValueError, match="No data found"):
        fetch_request_data(999999)
