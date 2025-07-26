import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_name: str
    openai_api_url: str
    openai_api_key: str
    save_dir: str = Field(default="results")
    save_step: int = Field(default=100)
    no_think: bool = Field(default=False)

    model_config = SettingsConfigDict(env_file=os.path.join(os.path.dirname(__file__), ".env"))
