from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://bhoonidhi-api.nrsc.gov.in"
LISS4_COLLECTIONS = [
    "ResourceSat-2_LISS4-MX70_L2",
    "ResourceSat-2A_LISS4-MX70_L2",
]


class BhoonidhiError(RuntimeError):
    """Raised when the Bhoonidhi API returns an error response."""


@dataclass
class TokenSet:
    user_id: str
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"

    @property
    def authorization_header(self) -> dict[str, str]:
        return {"Authorization": f"{self.token_type} {self.access_token}"}


class BhoonidhiClient:
    """Client for the Bhoonidhi STAC catalogue and download endpoints.

    Includes basic retry/backoff handling for rate limiting (HTTP 429)
    and transient server errors (HTTP 5xx).
    """

    def __init__(self, base_url: str = BASE_URL, timeout: int = 120, max_retries: int = 4) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    # ------------------------------------------------------------------ auth
    def authenticate(self, user_id: str, password: str) -> TokenSet:
        payload = {"userId": user_id, "password": password, "grant_type": "password"}
        return self._token_from_response(self._post("/auth/token", json=payload))

    def refresh(self, user_id: str, refresh_token: str) -> TokenSet:
        payload = {"userId": user_id, "refresh_token": refresh_token, "grant_type": "refresh_token"}
        return self._token_from_response(self._post("/auth/token", json=payload))

    @staticmethod
    def _token_from_response(data: dict[str, Any]) -> TokenSet:
        return TokenSet(
            user_id=data["userId"],
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            token_type=data.get("token_type", "Bearer"),
        )

    # ---------------------------------------------------------------- search
    def search(
        self,
        token: TokenSet,
        collections: list[str],
        datetime_range: str,
        bbox: tuple[float, float, float, float] | None = None,
        intersects: dict[str, Any] | None = None,
        limit: int = 100,
        online_only: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "collections": collections,
            "datetime": datetime_range,
            "limit": min(limit, 500),
        }
        if intersects is not None:
            payload["intersects"] = intersects
        elif bbox is not None:
            payload["bbox"] = [str(value) for value in bbox]
        if online_only:
            payload["filter"] = {"args": [{"property": "Online"}, "Y"], "op": "eq"}
            payload["filter-lang"] = "cql2-json"

        return self._post("/data/search", json=payload, headers=token.authorization_header)

    def search_all(
        self,
        token: TokenSet,
        collections: list[str],
        datetime_range: str,
        bbox: tuple[float, float, float, float] | None = None,
        intersects: dict[str, Any] | None = None,
        page_limit: int = 100,
        max_items: int = 2000,
        online_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Search and follow STAC pagination links until exhausted."""
        response = self.search(
            token=token,
            collections=collections,
            datetime_range=datetime_range,
            bbox=bbox,
            intersects=intersects,
            limit=page_limit,
            online_only=online_only,
        )
        features: list[dict[str, Any]] = list(response.get("features", []))

        while len(features) < max_items:
            next_link = _find_next_link(response.get("links", []))
            if next_link is None:
                break
            time.sleep(0.4)  # respect the 3 requests/second search limit
            response = self._follow_link(next_link, token)
            page_features = response.get("features", [])
            if not page_features:
                break
            features.extend(page_features)

        return features[:max_items]

    # -------------------------------------------------------------- download
    def download(
        self,
        token: TokenSet,
        item_id: str,
        collection: str,
        output_dir: Path,
        skip_existing: bool = True,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = _existing_download(output_dir, item_id)
        if skip_existing and existing is not None:
            return existing

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    f"{self.base_url}/download",
                    params={"id": item_id, "collection": collection},
                    headers=token.authorization_header,
                    timeout=self.timeout,
                    stream=True,
                )
                if response.status_code in (429, 412, 500, 502, 503, 504):
                    response.close()
                    time.sleep(_backoff_seconds(attempt))
                    continue
                self._raise_for_status(response)

                filename = _filename_from_response(response, fallback=f"{item_id}.zip")
                output_path = output_dir / filename
                partial_path = output_path.with_suffix(output_path.suffix + ".part")
                with partial_path.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
                partial_path.replace(output_path)
                return output_path
            except (requests.RequestException, BhoonidhiError) as exc:
                last_error = exc
                time.sleep(_backoff_seconds(attempt))

        raise BhoonidhiError(f"Failed to download {item_id} after {self.max_retries} attempts: {last_error}")

    def batch_download(
        self,
        token: TokenSet,
        features: Iterable[dict[str, Any]],
        output_dir: Path,
        skip_existing: bool = True,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[dict[str, Any]]:
        feature_list = list(features)
        results: list[dict[str, Any]] = []
        total = len(feature_list)
        for index, feature in enumerate(feature_list, start=1):
            item_id = feature.get("id", "")
            collection = feature.get("collection", "")
            record = {"id": item_id, "collection": collection, "status": "", "path": "", "error": ""}
            try:
                path = self.download(token, item_id, collection, output_dir, skip_existing=skip_existing)
                record["status"] = "downloaded"
                record["path"] = str(path)
            except Exception as exc:  # noqa: BLE001 - record per-item failure, continue batch.
                record["status"] = "failed"
                record["error"] = str(exc)
            results.append(record)
            if progress_callback is not None:
                progress_callback(index, total, item_id)
        return results

    # --------------------------------------------------------------- helpers
    def _follow_link(self, link: dict[str, Any], token: TokenSet) -> dict[str, Any]:
        href = link.get("href", "")
        method = str(link.get("method", "GET")).upper()
        if method == "POST":
            return self._request("POST", href, json=link.get("body", {}), headers=token.authorization_header)
        return self._request("GET", href, headers=token.authorization_header)

    def _post(
        self,
        endpoint: str,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return self._request("POST", f"{self.base_url}{endpoint}", json=json, headers=headers)

    def _request(
        self,
        method: str,
        url: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method,
                    url,
                    json=json,
                    headers=headers,
                    timeout=self.timeout,
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(_backoff_seconds(attempt))
                    last_error = BhoonidhiError(f"Transient error {response.status_code}")
                    continue
                self._raise_for_status(response)
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(_backoff_seconds(attempt))
        raise BhoonidhiError(f"Request to {url} failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        if response.ok:
            return
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise BhoonidhiError(f"Bhoonidhi API error {response.status_code}: {detail}")


def build_datetime_range(start_date: str, end_date: str) -> str:
    return f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"


def summarize_feature(feature: dict[str, Any]) -> dict[str, Any]:
    properties = feature.get("properties", {})
    return {
        "id": feature.get("id", ""),
        "collection": feature.get("collection", ""),
        "datetime": properties.get("datetime", properties.get("Acquisition_Date", "")),
        "online": properties.get("Online", properties.get("online", "")),
        "cloud": properties.get("Cloud_Cover", properties.get("Cloud_Percentage", properties.get("cloud_cover", ""))),
        "path": properties.get("Path", properties.get("path", "")),
        "row": properties.get("Row", properties.get("row", "")),
        "product": properties.get("Product_Type", properties.get("productType", "")),
    }


def _find_next_link(links: list[dict[str, Any]]) -> dict[str, Any] | None:
    for link in links:
        if str(link.get("rel", "")).lower() == "next" and link.get("href"):
            return link
    return None


def _existing_download(output_dir: Path, item_id: str) -> Path | None:
    for candidate in output_dir.glob(f"{item_id}*"):
        if candidate.is_file() and not candidate.name.endswith(".part"):
            return candidate
    return None


def _backoff_seconds(attempt: int) -> float:
    return min(2.0 ** attempt, 30.0)


def _filename_from_response(response: requests.Response, fallback: str) -> str:
    disposition = response.headers.get("content-disposition", "")
    for part in disposition.split(";"):
        part = part.strip()
        if part.lower().startswith("filename="):
            return part.split("=", 1)[1].strip('"')
    return fallback
