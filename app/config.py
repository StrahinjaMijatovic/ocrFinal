from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    docs_base_path: str = r"C:\Users\strahinja\Desktop\Test"
    db_server: str = "localhost"
    db_name: str = "NoviVis"
    db_driver: str = "ODBC Driver 17 for SQL Server"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    # Comma-separated PaddleOCR recognition models tried per image. The pipeline is
    # language-agnostic: every model runs and the highest-confidence result wins.
    # "latin" = English/Romanian/Serbian-latin, "cyrillic" = Serbian-cyrillic, "ch" = Chinese.
    ocr_languages: str = "latin,cyrillic,ch"
    # Filenames (without extension) skipped before any OCR. In every request folder
    # document "1" is always the applicant passport and "3" is an unneeded image, so
    # dropping them up front avoids wasted OCR + LLM work. Comma-separated.
    skip_document_filenames: str = "1,3"
    # Optional vision-model fallback: when PaddleOCR confidence falls below the
    # threshold, re-read the document with the vision model (higher quality on poor
    # scans, but much slower on CPU). Off by default.
    vision_fallback_enabled: bool = True
    vision_model: str = "qwen2.5vl:7b"
    ocr_confidence_threshold: float = 0.5
    active_structuring_prompt: str = "v2_few_shot"
    active_categorization_prompt: str = "v1_basic"
    # When a job finishes, its result (request id + extracted documents + categories)
    # is appended to this JSON file so completed analyses can be inspected/collected
    # in one place. Set to an empty string to disable file saving.
    results_file_path: str = r"runs\analyze_results.json"
    sla_warning_days: int = 5
    sla_breach_days: int = 10

    model_config = {"env_file": ".env"}


settings = Settings()
