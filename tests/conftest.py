import os
import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load environment variables from .env file before tests are run."""
    load_dotenv()

    # Verify that required API keys are available
    api_keys = ["OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    for key in api_keys:
        assert os.environ.get(key), f"Environment variable {key} is not set"
