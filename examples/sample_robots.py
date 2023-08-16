
from pydantic import BaseModel

# BaseMemory class (simplified with initialization)
class BaseMemory:
    def __init__(self, purpose: str):
        self.input_model = None
        self.output = ""
        self.purpose = purpose

# =========================================
# 1. Language Translation Robot
# =========================================
class TranslationRequest(BaseModel):
    source_language: str
    target_language: str
    content: str

class TranslationMemory(BaseMemory):
    def __init__(self, purpose: str):
        super().__init__(purpose)
        self.input_model = TranslationRequest(source_language="", target_language="", content="")
        self.instructions_for_ai = ""
        self.translated_content = ""

# =========================================
# 2. Weather Reporting Robot
# =========================================
class WeatherRequest(BaseModel):
    location: str

class WeatherMemory(BaseMemory):
    def __init__(self, purpose: str):
        super().__init__(purpose)
        self.input_model = WeatherRequest(location="")
        self.instructions_for_ai = ""
        self.weather_report = ""

# =========================================
# 3. Crypto Price Checker Robot
# =========================================
class CryptoRequest(BaseModel):
    cryptocurrency: str

class CryptoMemory(BaseMemory):
    def __init__(self, purpose: str):
        super().__init__(purpose)
        self.input_model = CryptoRequest(cryptocurrency="")
        self.instructions_for_ai = ""
        self.crypto_price = ""

# =========================================
# 4. Web Scraper Robot
# =========================================
class WebScrapeRequest(BaseModel):
    url: str

class WebScrapeMemory(BaseMemory):
    def __init__(self, purpose: str):
        super().__init__(purpose)
        self.input_model = WebScrapeRequest(url="")
        self.instructions_for_ai = ""
        self.scraped_data = ""

# ... [Rest of the code, including AIRobot class and execution logic]
