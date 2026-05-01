from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENV: str = "development"  # development | staging | production
    DB_SERVER: str = "localhost"
    DB_NAME: str = "SwishVision"
    DB_DRIVER: str = "ODBC Driver 17 for SQL Server"
    DB_TRUSTED_CONNECTION: bool = True
    DB_USERNAME: str = ""
    DB_PASSWORD: str = ""
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    SECRET_KEY: str = "change-me-to-a-random-256-bit-secret"
    ADMIN_RESET_KEY: str = ""
    UPLOAD_DIR: str = "app/uploads"
    OUTPUT_DIR: str = "output_videos"
    MODEL_DIR: str = "models"
    MAX_UPLOAD_SIZE_MB: int = 500
    ENV: str = "development"  # set to "production" in prod .env

    @property
    def DATABASE_URL(self) -> str:
        from urllib.parse import quote_plus
        if self.DB_TRUSTED_CONNECTION:
            params = quote_plus(
                f"DRIVER={{{self.DB_DRIVER}}};SERVER={self.DB_SERVER};"
                f"DATABASE={self.DB_NAME};Trusted_Connection=yes;"
            )
        else:
            params = quote_plus(
                f"DRIVER={{{self.DB_DRIVER}}};SERVER={self.DB_SERVER};"
                f"DATABASE={self.DB_NAME};UID={self.DB_USERNAME};PWD={self.DB_PASSWORD};"
            )
        return f"mssql+pyodbc:///?odbc_connect={params}"

    class Config:
        env_file = ".env"

settings = Settings()
