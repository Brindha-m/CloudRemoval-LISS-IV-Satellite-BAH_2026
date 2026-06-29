"""Area-of-interest definitions for the North Eastern Region (NER) of India.

Bounding boxes are approximate administrative extents in WGS84 degrees,
formatted as (min_lon, min_lat, max_lon, max_lat).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    name: str
    bbox: tuple[float, float, float, float]

    @property
    def polygon_coordinates(self) -> list[list[list[float]]]:
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return [
            [
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]
        ]

    @property
    def geojson(self) -> dict:
        return {"type": "Polygon", "coordinates": self.polygon_coordinates}


NER_STATES: dict[str, Region] = {
    "Arunachal Pradesh": Region("Arunachal Pradesh", (91.60, 26.60, 97.50, 29.55)),
    "Assam": Region("Assam", (89.65, 24.10, 96.05, 28.05)),
    "Manipur": Region("Manipur", (92.90, 23.80, 94.80, 25.70)),
    "Meghalaya": Region("Meghalaya", (89.80, 25.00, 92.85, 26.15)),
    "Mizoram": Region("Mizoram", (92.20, 21.90, 93.55, 24.55)),
    "Nagaland": Region("Nagaland", (93.30, 25.15, 95.25, 27.05)),
    "Tripura": Region("Tripura", (91.10, 22.90, 92.35, 24.55)),
    "Sikkim": Region("Sikkim", (88.00, 27.00, 88.95, 28.15)),
}

# Combined bounding box covering all eight NER states.
NER_BBOX: tuple[float, float, float, float] = (87.90, 21.90, 97.55, 29.60)

NER_FULL = Region("North Eastern Region", NER_BBOX)


def get_region(name: str) -> Region:
    if name == NER_FULL.name:
        return NER_FULL
    return NER_STATES[name]


def all_region_names() -> list[str]:
    return [NER_FULL.name, *NER_STATES.keys()]
