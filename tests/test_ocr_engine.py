from app.pipeline.ocr_engine import is_identity_mrz, is_photo_only


def test_identity_mrz_detects_passport():
    text = (
        "REPUBLIC OF CHINA\n"
        "P<CHNZHANG<<NANA<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
        "E759035120CHN0103015F2705136<<<<<<<<<<<<<<00"
    )
    assert is_identity_mrz(text) is True


def test_identity_mrz_ignores_visa():
    # A visa MRZ starts with 'V' — it is NOT an identity document, so it is kept.
    text = (
        "REPUBLIKA SRBIJA - VISA\n"
        "V<SRBALSALAHAT<<MUNEER<SOUD<SALIH<<<<<<<<<<<\n"
        "0780014248XXA5901010M1306145<<<<<<<<<<<<<<<"
    )
    assert is_identity_mrz(text) is False


def test_identity_mrz_false_on_plain_text():
    assert is_identity_mrz("This is an invitation letter with no MRZ zone.") is False
    # A document merely mentioning a passport number must NOT be flagged.
    assert is_identity_mrz("Passport number: E75903512 issued in Beijing.") is False


def test_is_photo_only_true_on_empty_text():
    assert is_photo_only("") is True
    assert is_photo_only("   \n  ") is True


def test_is_photo_only_false_on_real_text():
    assert is_photo_only("INVITATION LETTER\nDear Sir") is False
