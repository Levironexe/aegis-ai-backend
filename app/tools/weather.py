"""Weather Tool - Get current weather information using Open-Meteo API"""

import httpx
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from .base import BaseTool


class WeatherInput(BaseModel):
    """Input schema for weather tool."""
    latitude: Optional[float] = Field(None, description="Latitude of the location")
    longitude: Optional[float] = Field(None, description="Longitude of the location")
    city: Optional[str] = Field(None, description="City name for geocoding")


class WeatherTool(BaseTool):
    """Tool to get current weather at a location."""

    def __init__(self):
        super().__init__()
        self.name = "getWeather"
        self.needs_approval = True  # Requires approval before execution

    @property
    def description(self) -> str:
        return "Get the current weather at a location. Provide either coordinates (latitude, longitude) or a city name."

    @property
    def input_schema(self) -> type[BaseModel]:
        return WeatherInput

    async def _geocode_city(self, city: str) -> tuple[float, float]:
        """Convert city name to coordinates using Open-Meteo Geocoding API."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en", "format": "json"}
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("results"):
                raise ValueError(f"City '{city}' not found")

            result = data["results"][0]
            return result["latitude"], result["longitude"]

    async def execute(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        city: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute weather lookup.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            city: City name (alternative to coordinates)

        Returns:
            Weather data including temperature, hourly forecast, and daily info
        """
        # If city provided, geocode it first
        if city and (latitude is None or longitude is None):
            latitude, longitude = await self._geocode_city(city)

        if latitude is None or longitude is None:
            raise ValueError("Must provide either coordinates or city name")

        # Fetch weather data from Open-Meteo API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m",
                    "hourly": "temperature_2m",
                    "daily": "sunrise,sunset",
                    "timezone": "auto"
                }
            )
            response.raise_for_status()
            data = response.json()

        return {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": data.get("timezone"),
            "current": {
                "temperature": data.get("current", {}).get("temperature_2m"),
                "time": data.get("current", {}).get("time"),
            },
            "hourly": {
                "temperature": data.get("hourly", {}).get("temperature_2m", [])[:24],  # Next 24 hours
                "time": data.get("hourly", {}).get("time", [])[:24],
            },
            "daily": {
                "sunrise": data.get("daily", {}).get("sunrise", []),
                "sunset": data.get("daily", {}).get("sunset", []),
            }
        }
