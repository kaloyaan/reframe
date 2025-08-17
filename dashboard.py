#!/usr/bin/env python3

import os
import json
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
import httpx

CAMERA_AVAILABLE = False  # Dashboard no longer owns hardware; main service does

# Constants
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
PHOTOS_PATH = os.path.join(BASE_PATH, "photos")
DITHERED_PHOTOS_PATH = os.path.join(BASE_PATH, "dithered_photos")
SETTINGS_PATH = os.path.join(BASE_PATH, "settings.json")

# Ensure required directories exist
os.makedirs(PHOTOS_PATH, exist_ok=True)
os.makedirs(DITHERED_PHOTOS_PATH, exist_ok=True)

app = FastAPI(title="Reframe Dashboard", description="Control & Gallery Interface for Reframe Camera")

class SettingsManager:
    """Manages settings operations for the dashboard."""
    
    def __init__(self, settings_path: str = SETTINGS_PATH):
        self.settings_path = Path(settings_path)
        self.default_settings = {
            "camera": {
                "resolution": {"width": 1200, "height": 800},
                "exposure_value": -0.25,
                "sharpness": 3,
                "autofocus_mode": 2
            },
            "processing": {
                "saturation": 0.6,
                "brightness_factor": 1.1,
                "color_factor": 1.4,
                "dithering_method": "floyd_steinberg",
                "bayer_size": 4,
                "threshold_scale": 1.0
            },
            "display": {
                "auto_display": True,
                "display_timeout": 0
            },
            "system": {
                "auto_refresh_interval": 30
            }
        }
        self._ensure_settings_file()
    
    def _ensure_settings_file(self):
        """Ensure settings file exists with default values."""
        if not self.settings_path.exists():
            self.save_settings(self.default_settings)
    
    def load_settings(self) -> Dict[str, Any]:
        """Load settings from JSON file."""
        try:
            with open(self.settings_path, 'r') as f:
                settings = json.load(f)
            # Ensure all default keys exist
            return self._merge_with_defaults(settings)
        except (FileNotFoundError, json.JSONDecodeError):
            return self.default_settings.copy()
    
    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """Save settings to JSON file."""
        try:
            # Merge with existing settings to preserve structure
            current_settings = self.load_settings()
            merged_settings = self._deep_merge(current_settings, settings)
            
            with open(self.settings_path, 'w') as f:
                json.dump(merged_settings, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def _merge_with_defaults(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Merge loaded settings with defaults to ensure all keys exist."""
        return self._deep_merge(self.default_settings.copy(), settings)
    
    def _deep_merge(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get_camera_settings(self) -> Dict[str, Any]:
        """Get camera-specific settings."""
        return self.load_settings().get("camera", {})
    
    def get_processing_settings(self) -> Dict[str, Any]:
        """Get processing-specific settings."""
        return self.load_settings().get("processing", {})


class ReframeClient:
    """HTTP client to talk to the main reframe hardware service."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(30.0)

    async def get(self, path: str):
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

    async def post(self, path: str, json: Optional[Dict[str, Any]] = None):
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=json)
            resp.raise_for_status()
            return resp.json()

class PhotoManager:
    """Manages photo operations for the dashboard."""
    
    def __init__(self):
        self.photos_path = Path(PHOTOS_PATH)
        self.dithered_path = Path(DITHERED_PHOTOS_PATH)
        
    def get_all_photos(self, page: int = 1, limit: int = 20) -> Dict[str, Any]:
        """Get paginated list of photos with metadata."""
        all_photos = []
        
        # Get all files from photos directory
        if self.photos_path.exists():
            for photo_file in sorted(self.photos_path.iterdir(), reverse=True):
                if photo_file.is_file() and photo_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Look for corresponding dithered version
                    dithered_file = self.dithered_path / f"{photo_file.stem}_dithered{photo_file.suffix}"
                    if not dithered_file.exists():
                        # Try without _dithered suffix for exact matches
                        dithered_file = self.dithered_path / photo_file.name
                    
                    photo_info = {
                        "id": photo_file.stem,
                        "filename": photo_file.name,
                        "original_path": f"/photos/{photo_file.name}",
                        "dithered_path": f"/dithered/{dithered_file.name}" if dithered_file.exists() else None,
                        "has_dithered": dithered_file.exists(),
                        "size": photo_file.stat().st_size,
                        "created": datetime.fromtimestamp(photo_file.stat().st_mtime).isoformat()
                    }
                    all_photos.append(photo_info)
        
        # Calculate pagination
        total_photos = len(all_photos)
        total_pages = (total_photos + limit - 1) // limit  # Ceiling division
        start_index = (page - 1) * limit
        end_index = start_index + limit
        
        # Get photos for current page
        photos_page = all_photos[start_index:end_index]
        
        return {
            "photos": photos_page,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "total_photos": total_photos,
                "photos_per_page": limit,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        }
    
    def get_photo_info(self, photo_id: str) -> Dict:
        """Get information about a specific photo."""
        # Get all photos without pagination to search through them
        all_photos_data = self.get_all_photos(page=1, limit=10000)  # Large limit to get all
        photos = all_photos_data["photos"]
        for photo in photos:
            if photo["id"] == photo_id:
                return photo
        raise HTTPException(status_code=404, detail="Photo not found")

# Initialize managers
settings_manager = SettingsManager()
photo_manager = PhotoManager()

# Initialize HTTP client to the hardware service
REFRAME_API_BASE = os.environ.get("REFRAME_API_BASE", "http://127.0.0.1:8077/api")
reframe_client = ReframeClient(REFRAME_API_BASE)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard interface."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reframe</title>
        <style>
            :root {
                --primary-color: #F5F1F0;  /* background */
                --secondary-color: #181818;             /* text and borders */
                --tertiary-color: #F5F1F0;              /* content backgrounds */
                --hover-color: #333;                  /* hover state color */
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: serif;
                background: var(--primary-color);
                min-height: 100vh;
                color: var(--secondary-color);
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 40px;
            }
            
            .header {
                text-align: left;
                margin-bottom: 60px;
            }
            
            .header h1 {
                color: var(--secondary-color);
                font-size: 1rem;
                font-weight: normal;
            }
            
            .header p {
                color: var(--secondary-color);
                font-size: 1rem;
                opacity: 0.8;
            }
            
            .controls {
                background: var(--tertiary-color);
                padding: 40px;
                margin-bottom: 40px;
            }
            
            .button {
                background: var(--secondary-color);
                color: var(--tertiary-color);
                border: none;
                padding: 5px 10px;
                cursor: pointer;
                font-size: 1rem;
                font-family: serif;
            }
            
            .button:hover {
                background: var(--hover-color);
            }
            
            .gallery {
                /* background: white; */
                /* padding: 40px; */
            }
            
            .pagination {
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 10px;
                margin: 40px 0;
                padding: 20px;
            }
            
            .pagination-btn {
                background: var(--tertiary-color);
                color: var(--secondary-color);
                border: 1px solid var(--secondary-color);
                padding: 5px 10px;
                cursor: pointer;
                font-family: serif;
                font-size: 1rem;
                min-width: 40px;
                text-align: center;
            }
            
            .pagination-btn:hover:not(:disabled) {
                background: var(--secondary-color);
                color: var(--tertiary-color);
            }
            
            .pagination-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            .pagination-btn.active {
                background: var(--secondary-color);
                color: var(--tertiary-color);
            }
            
            .pagination-info {
                font-family: serif;
                font-size: 1rem;
                margin: 0 20px;
            }
            
            .gallery h2 {
                margin-bottom: 40px;
                color: var(--secondary-color);
                border-bottom: 4px solid var(--secondary-color);
                padding-bottom: 20px;
                font-size: 1rem;
                font-weight: normal;
            }
            
            .photo-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
            }
            
            .photo-card {
                border: 2px solid var(--secondary-color);
                overflow: hidden;
                background: var(--tertiary-color);
                position: relative;
            }
                        
            .photo-image {
                width: 100%;
                object-fit: cover;
                cursor: pointer;
                display: block;
            }
            
            .photo-info {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                background: rgba(0, 0, 0, 0.7);
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.3s ease;
                padding: 10px;
            }
            
            .photo-card:hover .photo-info {
                opacity: 1;
                pointer-events: auto;
            }
            
            .photo-card.active .photo-info {
                opacity: 1;
                pointer-events: auto;
            }
            
            @media (hover: none) and (pointer: coarse) {
                .photo-card:hover .photo-info {
                    opacity: 0;
                    pointer-events: none;
                }
                
                .photo-card.active .photo-info {
                    opacity: 1;
                    pointer-events: auto;
                }
            }
            
            .photo-actions {
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
                justify-content: center;
            }
            
            .action-btn {
                padding: 8px 12px;
                font-size: 0.9rem;
                text-decoration: none;
                font-family: serif;
                border: 1px solid var(--tertiary-color);
                background: rgba(255, 255, 255, 0.9);
                color: var(--secondary-color);
                transition: all 0.2s ease;
                display: inline-flex;
                align-items: center;
                backdrop-filter: blur(2px);
            }
            
            .btn-primary {
                background: rgba(255, 255, 255, 0.9);
                color: var(--secondary-color);
            }
            
            .btn-secondary {
                background: rgba(255, 255, 255, 0.9);
                color: var(--secondary-color);
            }
            
            .btn-success {
                background: rgba(255, 255, 255, 0.9);
                color: var(--secondary-color);
            }
            
            .action-btn:hover {
                background: var(--tertiary-color);
                color: var(--secondary-color);
                transform: translateY(-1px);
            }
            
            .loading {
                text-align: center;
                padding: 60px;
                color: var(--secondary-color);
                font-size: 1rem;
            }
            
            .status-bar {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .status-item {
                display: flex;
                align-items: center;
                gap: 15px;
                font-size: 1rem;
            }
            
            .status-indicator {
                width: 12px;
                height: 12px;
                background: var(--secondary-color);
            }
            
            .settings-modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.8);
            }
            
            .settings-content {
                background-color: var(--tertiary-color);
                margin: 5% auto;
                padding: 40px;
                border: 2px solid var(--secondary-color);
                width: 90%;
                max-width: 800px;
                max-height: 80vh;
                overflow-y: auto;
            }
            
            .settings-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 30px;
                border-bottom: 2px solid var(--secondary-color);
                padding-bottom: 20px;
            }
            
            .settings-header h2 {
                font-size: 1rem;
                font-weight: normal;
            }
            
            .close-btn {
                background: var(--secondary-color);
                color: var(--tertiary-color);
                border: none;
                padding: 10px 20px;
                cursor: pointer;
                font-family: serif;
                font-size: 1rem;
            }
            
            .close-btn:hover {
                background: var(--hover-color);
            }
            
            .settings-section {
                margin-bottom: 30px;
                border: 1px solid var(--secondary-color);
                padding: 20px;
            }
            
            .settings-section h3 {
                font-size: 1rem;
                margin-bottom: 20px;
                font-weight: normal;
                border-bottom: 1px solid var(--secondary-color);
                padding-bottom: 10px;
            }
            
            .setting-group {
                margin-bottom: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 10px;
            }
            
            .setting-label {
                font-weight: bold;
                min-width: 200px;
            }
            
            .setting-input {
                border: 1px solid var(--secondary-color);
                padding: 8px 12px;
                font-family: serif;
                font-size: 1rem;
                min-width: 100px;
            }
            
            .setting-input:focus {
                /* outline: 2px solid var(--secondary-color); */
            }
            
            .disabled-setting {
                opacity: 0.5;
                pointer-events: none;
            }
            
            .disabled-setting .setting-label {
                color: #999;
            }
            
            .disabled-setting .setting-input {
                background-color: #f5f5f5;
                color: #999;
                cursor: not-allowed;
            }
            
            #bayer-settings, #threshold-settings {
                display: none;
            }
            
            .settings-actions {
                margin-top: 30px;
                text-align: center;
                border-top: 2px solid var(--secondary-color);
                padding-top: 20px;
            }
            
            .save-btn {
                background: var(--secondary-color);
                color: var(--tertiary-color);
                border: none;
                padding: 15px 30px;
                cursor: pointer;
                font-family: serif;
                font-size: 1rem;
                margin-right: 20px;
            }
            
            .save-btn:hover {
                background: var(--hover-color);
            }
            
            .reset-btn {
                background: var(--tertiary-color);
                color: var(--secondary-color);
                border: 1px solid var(--secondary-color);
                padding: 15px 30px;
                cursor: pointer;
                font-family: serif;
                font-size: 1rem;
            }
            
            .reset-btn:hover {
                background: var(--secondary-color);
                color: var(--tertiary-color);
            }
            
            @media (max-width: 768px) {
                .photo-grid {
                    grid-template-columns: 1fr;
                }
                
                .header h1 {
                    font-size: 1rem;
                }
                
                .container {
                    padding: 20px;
                }
                
                .status-bar {
                    flex-direction: row;
                    flex-wrap: wrap;
                    gap: 20px;
                    text-align: center;
                }
                
                .settings-content {
                    margin: 2% auto;
                    padding: 20px;
                    width: 95%;
                }
                
                .setting-group {
                    flex-direction: column;
                    align-items: flex-start;
                }
                
                .setting-label {
                    min-width: auto;
                }
                
                .pagination {
                    flex-wrap: wrap;
                    gap: 5px;
                }
                
                .pagination-info {
                    margin: 10px 0;
                    width: 100%;
                    text-align: center;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="status-bar">
                <h1>reframe.camera dashboard</h1>
                <div class="status-item">
                    <div class="status-indicator"></div>
                    <span>system online</span>
                </div>
                <div class="status-item">
                    <span id="photo-count">loading photos...</span>
                </div>
                <div class="status-item">
                    <button class="button" onclick="refreshGallery()">refresh</button>
                </div>
                <button class="button" onclick="capturePhoto()">capture photo</button>
                <button class="button" onclick="openSettings()">settings</button>
                <button class="button" onclick="clearScreen()">clear screen</button>
            </div>
            </div>
            
            <div class="gallery">
                <div id="photo-grid" class="photo-grid">
                    <div class="loading">loading photos...</div>
                </div>
                
                <div id="pagination" class="pagination" style="display: none;">
                    <button class="pagination-btn" id="prev-btn" onclick="changePage(currentPage - 1)">previous</button>
                    <div id="page-numbers"></div>
                    <div class="pagination-info">
                        <span id="pagination-info"></span>
                    </div>
                    <button class="pagination-btn" id="next-btn" onclick="changePage(currentPage + 1)">next</button>
                </div>
            </div>
        </div>
        
        <!-- Settings Modal -->
        <div id="settings-modal" class="settings-modal">
            <div class="settings-content">
                <div class="settings-header">
                    <h2>settings</h2>
                    <button class="close-btn" onclick="closeSettings()">close</button>
                </div>
                
                <div class="settings-section">
                    <h3>camera settings</h3>
                    <div class="setting-group disabled-setting">
                        <span class="setting-label">Resolution Width:</span>
                        <input type="number" id="resolution-width" class="setting-input" min="100" max="4000" disabled>
                    </div>
                    <div class="setting-group disabled-setting">
                        <span class="setting-label">Resolution Height:</span>
                        <input type="number" id="resolution-height" class="setting-input" min="100" max="4000" disabled>
                    </div>
                    <div class="setting-group">
                        <span class="setting-label">Exposure Value:</span>
                        <input type="number" id="exposure-value" class="setting-input" step="0.25" min="-2" max="2">
                    </div>
                    <div class="setting-group">
                        <span class="setting-label">Sharpness:</span>
                        <input type="number" id="sharpness" class="setting-input" min="0" max="10">
                    </div>
                    <div class="setting-group">
                        <span class="setting-label">Autofocus Mode:</span>
                        <select id="autofocus-mode" class="setting-input">
                            <option value="0">Manual</option>
                            <option value="1">Auto</option>
                            <option value="2">Continuous</option>
                        </select>
                    </div>
                </div>
                
                <div class="settings-section">
                    <h3>processing settings</h3>
                    <div class="setting-group">
                        <span class="setting-label">Saturation:</span>
                        <input type="number" id="saturation" class="setting-input" step="0.1" min="0" max="2">
                    </div>
                    <div class="setting-group">
                        <span class="setting-label">Brightness Factor:</span>
                        <input type="number" id="brightness-factor" class="setting-input" step="0.1" min="0.1" max="3">
                    </div>
                    <div class="setting-group">
                        <span class="setting-label">Color Factor:</span>
                        <input type="number" id="color-factor" class="setting-input" step="0.1" min="0.1" max="3">
                    </div>
                    <div class="setting-group">
                        <span class="setting-label">dithering method:</span>
                        <select id="dithering-method" class="setting-input">
                            <option value="floyd_steinberg">floyd steinberg</option>
                            <option value="ordered">ordered (bayer)</option>
                        </select>
                    </div>
                    <div class="setting-group" id="bayer-settings">
                        <span class="setting-label">bayer matrix size:</span>
                        <select id="bayer-size" class="setting-input">
                            <option value="2">2x2</option>
                            <option value="4">4x4</option>
                            <option value="8">8x8</option>
                        </select>
                    </div>
                    <div class="setting-group" id="threshold-settings">
                        <span class="setting-label">threshold scale:</span>
                        <input type="number" id="threshold-scale" class="setting-input" min="0.1" max="2.0" step="0.1">
                    </div>
                </div>
                
                <div class="settings-section">
                    <h3>system settings</h3>
                    <div class="setting-group">
                        <span class="setting-label">Auto Refresh Interval (seconds):</span>
                        <input type="number" id="auto-refresh-interval" class="setting-input" min="5" max="300">
                    </div>

                </div>
                
                <div class="settings-actions">
                    <button class="save-btn" onclick="saveSettings()">save settings</button>
                    <button class="reset-btn" onclick="resetSettings()">reset to defaults</button>
                </div>
            </div>
        </div>
        
        <script>
            let photos = [];
            let pagination = {};
            let currentPage = 1;
            const photosPerPage = 12;  // Show 12 photos per page
            
            async function loadPhotos(page = 1) {
                try {
                    const response = await fetch(`/api/photos?page=${page}&limit=${photosPerPage}`);
                    const data = await response.json();
                    photos = data.photos;
                    pagination = data.pagination;
                    currentPage = page;
                    renderGallery();
                    updatePhotoCount();
                    renderPagination();
                } catch (error) {
                    console.error('Error loading photos:', error);
                    document.getElementById('photo-grid').innerHTML = '<div class="loading">error loading photos</div>';
                }
            }
            
            async function clearScreen() {
                try {
                    const btn = document.querySelector('button[onclick="clearScreen()"]');
                    if (btn) btn.disabled = true;
                    const resp = await fetch('/api/display/clear', { method: 'POST' });
                    if (!resp.ok) throw new Error(await resp.text());
                    const data = await resp.json();
                    alert(data.message || 'screen cleared');
                } catch (error) {
                    console.error('Error clearing screen:', error);
                    alert('failed to clear screen');
                } finally {
                    const btn = document.querySelector('button[onclick="clearScreen()"]');
                    if (btn) btn.disabled = false;
                }
            }

            function renderGallery() {
                const grid = document.getElementById('photo-grid');
                
                if (photos.length === 0) {
                    grid.innerHTML = '<div class="loading">no photos found. capture your first photo!</div>';
                    return;
                }
                
                grid.innerHTML = photos.map(photo => `
                    <div class="photo-card" data-photo-id="${photo.id}">
                        <img 
                            src="${photo.dithered_path || photo.original_path}" 
                            alt="Photo ${photo.id}"
                            class="photo-image"
                        />
                        <div class="photo-info">
                            <div class="photo-actions">
                                <a href="${photo.original_path}" class="action-btn btn-primary" download onclick="event.stopPropagation()">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 48 48" style="margin-right: 5px;">
                                        <path fill="currentColor" d="M26 6a2 2 0 1 0-4 0h4Zm-3.414 37.414a2 2 0 0 0 2.828 0l12.728-12.728a2 2 0 1 0-2.828-2.828L24 39.172 12.686 27.858a2 2 0 1 0-2.828 2.828l12.728 12.728ZM24 6h-2v36h4V6h-2Z"/>
                                    </svg>
                                    original
                                </a>
                                ${photo.has_dithered ? `
                                    <a href="${photo.dithered_path}" class="action-btn btn-secondary" download onclick="event.stopPropagation()">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 48 48" style="margin-right: 5px;">
                                            <path fill="currentColor" d="M10 28h4v4h-4v-4Zm4 4h4v4h-4v-4Z"/>
                                            <path fill="currentColor" d="M14 32h4v4h-4v-4Zm4 4h4v4h-4v-4Zm20-8h-4v4h4v-4Zm-4 4h-4v4h4v-4Zm-4 4h-4v4h4v-4Zm-8 4h4v4h-4v-4Zm0-4h4v4h-4v-4Zm0-4h4v4h-4v-4Zm0-4h4v4h-4v-4Zm0-4h4v4h-4v-4Zm0-4h4v4h-4v-4Zm0-4h4v4h-4v-4Zm0-4h4v4h-4v-4Zm0-4h4v4h-4V8Zm0-4h4v4h-4V4Z"/>
                                        </svg>
                                        dithered
                                    </a>
                                ` : ''}
                                <button class="action-btn btn-success" onclick="event.stopPropagation(); displayPhoto('${photo.id}')">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 48 48" style="margin-right: 5px;">
                                        <path fill="currentColor" d="M6 30h12v12H6V30Zm12-12h12v12H18V18ZM30 6h12v12H30V6Zm0 24h12v12H30V30ZM6 6h12v12H6V6Z"/>
                                    </svg>
                                    display
                                </button>
                            </div>
                        </div>
                    </div>
                `).join('');
                
                // Add click handlers for photo cards
                setupPhotoCardHandlers();
            }
            
            function updatePhotoCount() {
                if (pagination.total_photos !== undefined) {
                    document.getElementById('photo-count').textContent = `${pagination.total_photos} photos`;
                } else {
                    document.getElementById('photo-count').textContent = `${photos.length} photos`;
                }
            }
            
            function renderPagination() {
                const paginationDiv = document.getElementById('pagination');
                const pageNumbersDiv = document.getElementById('page-numbers');
                const paginationInfo = document.getElementById('pagination-info');
                const prevBtn = document.getElementById('prev-btn');
                const nextBtn = document.getElementById('next-btn');
                
                if (!pagination || pagination.total_pages <= 1) {
                    paginationDiv.style.display = 'none';
                    return;
                }
                
                paginationDiv.style.display = 'flex';
                
                // Update navigation buttons
                prevBtn.disabled = !pagination.has_prev;
                nextBtn.disabled = !pagination.has_next;
                
                // Update pagination info
                const startItem = ((currentPage - 1) * photosPerPage) + 1;
                const endItem = Math.min(currentPage * photosPerPage, pagination.total_photos);
                paginationInfo.textContent = `${startItem}-${endItem} of ${pagination.total_photos}`;
                
                // Generate page numbers
                pageNumbersDiv.innerHTML = '';
                const maxVisiblePages = 5;
                let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
                let endPage = Math.min(pagination.total_pages, startPage + maxVisiblePages - 1);
                
                // Adjust start page if we're near the end
                if (endPage - startPage + 1 < maxVisiblePages) {
                    startPage = Math.max(1, endPage - maxVisiblePages + 1);
                }
                
                // Add first page and ellipsis if needed
                if (startPage > 1) {
                    addPageButton(1);
                    if (startPage > 2) {
                        const ellipsis = document.createElement('span');
                        ellipsis.textContent = '...';
                        ellipsis.className = 'pagination-info';
                        pageNumbersDiv.appendChild(ellipsis);
                    }
                }
                
                // Add visible page numbers
                for (let i = startPage; i <= endPage; i++) {
                    addPageButton(i);
                }
                
                // Add last page and ellipsis if needed
                if (endPage < pagination.total_pages) {
                    if (endPage < pagination.total_pages - 1) {
                        const ellipsis = document.createElement('span');
                        ellipsis.textContent = '...';
                        ellipsis.className = 'pagination-info';
                        pageNumbersDiv.appendChild(ellipsis);
                    }
                    addPageButton(pagination.total_pages);
                }
            }
            
            function addPageButton(pageNum) {
                const button = document.createElement('button');
                button.textContent = pageNum;
                button.className = 'pagination-btn' + (pageNum === currentPage ? ' active' : '');
                button.onclick = () => changePage(pageNum);
                document.getElementById('page-numbers').appendChild(button);
            }
            
            function changePage(page) {
                if (page >= 1 && page <= pagination.total_pages && page !== currentPage) {
                    loadPhotos(page);
                }
            }
            
            function setupPhotoCardHandlers() {
                const photoCards = document.querySelectorAll('.photo-card');
                
                photoCards.forEach(card => {
                    let tapTimeout;
                    let lastTap = 0;
                    
                    // Handle click/tap events
                    card.addEventListener('click', function(e) {
                        const currentTime = new Date().getTime();
                        const tapLength = currentTime - lastTap;
                        
                        // Check if this is a touch device
                        const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
                        
                        if (isTouchDevice) {
                            // On mobile: first tap shows buttons, second tap (double-tap) opens photo
                            if (tapLength < 500 && tapLength > 0) {
                                // Double tap - open photo
                                const photoId = card.getAttribute('data-photo-id');
                                viewPhoto(photoId);
                                card.classList.remove('active');
                            } else {
                                // Single tap - toggle buttons
                                clearTimeout(tapTimeout);
                                tapTimeout = setTimeout(() => {
                                    // Remove active class from all other cards
                                    photoCards.forEach(otherCard => {
                                        if (otherCard !== card) {
                                            otherCard.classList.remove('active');
                                        }
                                    });
                                    // Toggle this card
                                    card.classList.toggle('active');
                                }, 300);
                            }
                            lastTap = currentTime;
                        } else {
                            // On desktop: single click opens photo (hover shows buttons)
                            const photoId = card.getAttribute('data-photo-id');
                            viewPhoto(photoId);
                        }
                    });
                    
                    // Close buttons when clicking outside on mobile
                    document.addEventListener('click', function(e) {
                        if (!card.contains(e.target)) {
                            card.classList.remove('active');
                        }
                    });
                });
            }
            
            function formatFileSize(bytes) {
                const units = ['B', 'KB', 'MB', 'GB'];
                let size = bytes;
                let unitIndex = 0;
                
                while (size >= 1024 && unitIndex < units.length - 1) {
                    size /= 1024;
                    unitIndex++;
                }
                
                return `${size.toFixed(1)} ${units[unitIndex]}`;
            }
            
            function formatDate(isoString) {
                return new Date(isoString).toLocaleDateString();
            }
            
            function viewPhoto(photoId) {
                const photo = photos.find(p => p.id === photoId);
                if (photo) {
                    // Show dithered version if available, otherwise show original
                    const imageToShow = photo.has_dithered ? photo.dithered_path : photo.original_path;
                    window.open(imageToShow, '_blank');
                }
            }
            
            async function displayPhoto(photoId) {
                try {
                    const response = await fetch(`/api/display/${photoId}`, {
                        method: 'POST'
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        alert(result.message);
                    } else {
                        const error = await response.json();
                        alert(`Error: ${error.detail}`);
                    }
                } catch (error) {
                    console.error('Error displaying photo:', error);
                    alert('Error displaying photo');
                }
            }
            
            async function capturePhoto() {
                try {
                    // Find the capture button and show loading indicator
                    const captureBtn = document.querySelector('button[onclick="capturePhoto()"]');
                    if (captureBtn) {
                        captureBtn.textContent = 'capturing...';
                        captureBtn.disabled = true;
                    }
                    
                    const response = await fetch('/api/capture', {
                        method: 'POST'
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        alert(result.message);
                        // Refresh gallery to show new photo (go to first page since new photos appear first)
                        loadPhotos(1);
                    } else {
                        const error = await response.json();
                        alert(`Error: ${error.detail}`);
                    }
                } catch (error) {
                    console.error('Error capturing photo:', error);
                    alert('Error capturing photo');
                } finally {
                    // Restore button state
                    const captureBtn = document.querySelector('button[onclick="capturePhoto()"]');
                    if (captureBtn) {
                        captureBtn.textContent = 'capture photo';
                        captureBtn.disabled = false;
                    }
                }
            }
            
            async function reprocessPhoto(photoId) {
                try {
                    const response = await fetch(`/api/reprocess/${photoId}`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        }
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        alert(result.message);
                        // Refresh gallery to show updated photo (stay on current page)
                        loadPhotos(currentPage);
                    } else {
                        const error = await response.json();
                        alert(`Error: ${error.detail}`);
                    }
                } catch (error) {
                    console.error('Error reprocessing photo:', error);
                    alert('Error reprocessing photo');
                }
            }
            
            async function openSettings() {
                try {
                    const response = await fetch('/api/settings');
                    const settings = await response.json();
                    populateSettingsForm(settings);
                    document.getElementById('settings-modal').style.display = 'block';
                } catch (error) {
                    console.error('Error loading settings:', error);
                    alert('Error loading settings');
                }
            }
            
            function closeSettings() {
                document.getElementById('settings-modal').style.display = 'none';
            }
            
            function populateSettingsForm(settings) {
                // Camera settings
                document.getElementById('resolution-width').value = settings.camera.resolution.width;
                document.getElementById('resolution-height').value = settings.camera.resolution.height;
                document.getElementById('exposure-value').value = settings.camera.exposure_value;
                document.getElementById('sharpness').value = settings.camera.sharpness;
                document.getElementById('autofocus-mode').value = settings.camera.autofocus_mode;
                
                // Processing settings
                document.getElementById('saturation').value = settings.processing.saturation;
                document.getElementById('brightness-factor').value = settings.processing.brightness_factor;
                document.getElementById('color-factor').value = settings.processing.color_factor;
                document.getElementById('dithering-method').value = settings.processing.dithering_method;
                document.getElementById('bayer-size').value = settings.processing.bayer_size || 4;
                document.getElementById('threshold-scale').value = settings.processing.threshold_scale || 1.0;
                
                // Show/hide ordered dithering settings
                toggleOrderedSettings();
                
                // System settings
                document.getElementById('auto-refresh-interval').value = settings.system.auto_refresh_interval;
            }
            
            async function saveSettings() {
                try {
                    const settings = {
                        camera: {
                            resolution: {
                                width: parseInt(document.getElementById('resolution-width').value),
                                height: parseInt(document.getElementById('resolution-height').value)
                            },
                            exposure_value: parseFloat(document.getElementById('exposure-value').value),
                            sharpness: parseInt(document.getElementById('sharpness').value),
                            autofocus_mode: parseInt(document.getElementById('autofocus-mode').value)
                        },
                        processing: {
                            saturation: parseFloat(document.getElementById('saturation').value),
                            brightness_factor: parseFloat(document.getElementById('brightness-factor').value),
                            color_factor: parseFloat(document.getElementById('color-factor').value),
                            dithering_method: document.getElementById('dithering-method').value,
                            bayer_size: parseInt(document.getElementById('bayer-size').value),
                            threshold_scale: parseFloat(document.getElementById('threshold-scale').value)
                        },
                        system: {
                                            auto_refresh_interval: parseInt(document.getElementById('auto-refresh-interval').value)
                        }
                    };
                    
                    const response = await fetch('/api/settings', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(settings)
                    });
                    
                    if (response.ok) {
                        alert('Settings saved successfully!');
                        closeSettings();
                        // Update auto-refresh interval if changed
                        updateAutoRefreshInterval();
                    } else {
                        alert('Error saving settings');
                    }
                } catch (error) {
                    console.error('Error saving settings:', error);
                    alert('Error saving settings');
                }
            }
            
            async function resetSettings() {
                if (confirm('Are you sure you want to reset all settings to defaults?')) {
                    try {
                        const defaultSettings = {
                            camera: {
                                resolution: {width: 1200, height: 800},
                                exposure_value: -0.25,
                                sharpness: 3,
                                autofocus_mode: 2
                            },
                            processing: {
                                saturation: 0.6,
                                brightness_factor: 1.1,
                                color_factor: 1.4,
                                dithering_method: "floyd_steinberg",
                                bayer_size: 4,
                                threshold_scale: 1.0
                            },
                            system: {
                                auto_refresh_interval: 30
                            }
                        };
                        
                        const response = await fetch('/api/settings', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify(defaultSettings)
                        });
                        
                        if (response.ok) {
                            populateSettingsForm(defaultSettings);
                            alert('Settings reset to defaults!');
                        } else {
                            alert('Error resetting settings');
                        }
                    } catch (error) {
                        console.error('Error resetting settings:', error);
                        alert('Error resetting settings');
                    }
                }
            }
            
            let autoRefreshInterval;
            
            function updateAutoRefreshInterval() {
                // Clear existing interval
                if (autoRefreshInterval) {
                    clearInterval(autoRefreshInterval);
                }
                
                // Get current auto-refresh setting
                fetch('/api/settings')
                    .then(response => response.json())
                    .then(settings => {
                        const intervalSeconds = settings.system.auto_refresh_interval;
                        if (intervalSeconds > 0) {
                            autoRefreshInterval = setInterval(loadPhotos, intervalSeconds * 1000);
                        }
                    })
                    .catch(error => console.error('Error updating auto-refresh:', error));
            }
            
            function toggleOrderedSettings() {
                const ditheringMethod = document.getElementById('dithering-method').value;
                const bayerSettings = document.getElementById('bayer-settings');
                const thresholdSettings = document.getElementById('threshold-settings');
                
                if (ditheringMethod === 'ordered') {
                    bayerSettings.style.display = 'block';
                    thresholdSettings.style.display = 'block';
                } else {
                    bayerSettings.style.display = 'none';
                    thresholdSettings.style.display = 'none';
                }
            }
            
            function refreshGallery() {
                loadPhotos(currentPage);
            }
            
            // Load photos on page load
            document.addEventListener('DOMContentLoaded', function() {
                loadPhotos();
                updateAutoRefreshInterval();
                
                // Add event listener for dithering method changes
                document.getElementById('dithering-method').addEventListener('change', toggleOrderedSettings);
            });
            
            // Close modal when clicking outside of it
            window.onclick = function(event) {
                const modal = document.getElementById('settings-modal');
                if (event.target === modal) {
                    closeSettings();
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/photos")
async def list_photos(page: int = 1, limit: int = 20):
    """Get paginated list of photos with metadata from the hardware service."""
    if page < 1:
        page = 1
    if limit < 1 or limit > 100:
        limit = 20
    # Fetch all photos from hardware service and paginate here for simplicity
    all_photos = await reframe_client.get("/photos")
    # Rewrite absolute file system paths to dashboard-served URLs
    for photo in all_photos:
        try:
            if photo.get("original_path"):
                from os.path import basename as _bn
                orig_name = _bn(photo["original_path"])
                photo["original_path"] = f"/photos/{orig_name}"
            if photo.get("dithered_path"):
                from os.path import basename as _bn
                dith_name = _bn(photo["dithered_path"])
                photo["dithered_path"] = f"/dithered/{dith_name}"
        except Exception:
            continue
    start = (page - 1) * limit
    end = start + limit
    total = len(all_photos)
    total_pages = (total + limit - 1) // limit if total else 1
    return {
        "photos": all_photos[start:end],
        "pagination": {
            "page": page,
            "limit": limit,
            "total_photos": total,
            "total_pages": total_pages,
            "has_prev": page > 1,
            "has_next": page < total_pages,
        },
    }

@app.get("/api/photos/{photo_id}")
async def get_photo_info(photo_id: str):
    """Get information about a specific photo from the hardware service."""
    try:
        photo = await reframe_client.get(f"/photos/{photo_id}")
        # Rewrite paths to URLs served by this dashboard
        from os.path import basename as _bn
        if photo.get("original_path"):
            photo["original_path"] = f"/photos/{_bn(photo['original_path'])}"
        if photo.get("dithered_path"):
            photo["dithered_path"] = f"/dithered/{_bn(photo['dithered_path'])}"
        return photo
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/photos/{filename}")
async def serve_original_photo(filename: str):
    """Serve original photo file."""
    file_path = os.path.join(PHOTOS_PATH, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Photo not found")
    return FileResponse(file_path)

@app.get("/dithered/{filename}")
async def serve_dithered_photo(filename: str):
    """Serve dithered photo file."""
    file_path = os.path.join(DITHERED_PHOTOS_PATH, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dithered photo not found")
    return FileResponse(file_path)

@app.get("/api/settings")
async def get_settings():
    """Get current settings."""
    return settings_manager.load_settings()

@app.post("/api/settings")
async def update_settings(request: Request):
    """Update settings and notify the hardware service to reload/apply."""
    try:
        settings_data = await request.json()
        success = settings_manager.save_settings(settings_data)
        if success:
            try:
                await reframe_client.post("/settings/reload")
            except Exception:
                # Hardware service might be down; still consider settings saved
                pass
            return {"status": "success", "message": "Settings updated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save settings")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid settings data: {str(e)}")

@app.get("/api/settings/camera")
async def get_camera_settings():
    """Get camera-specific settings."""
    return settings_manager.get_camera_settings()

@app.get("/api/settings/processing") 
async def get_processing_settings():
    """Get processing-specific settings."""
    return settings_manager.get_processing_settings()

@app.post("/api/capture")
async def capture_photo(background_tasks: BackgroundTasks):
    """Capture a new photo with current settings."""
    try:
        result = await reframe_client.post("/capture")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/display/clear")
async def clear_display():
    """Proxy to clear the e-ink display on the hardware service."""
    try:
        resp = await reframe_client.post("/display/clear")
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear display: {str(e)}")

@app.post("/api/display/{photo_id}")
async def display_photo_on_screen(photo_id: str):
    """Display a specific photo on the e-ink screen."""
    try:
        result = await reframe_client.post(f"/display/{photo_id}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reprocess/{photo_id}")
async def reprocess_photo(photo_id: str, request: Request):
    """Reprocess an existing photo with new settings."""
    try:
        processing_settings = None
        try:
            body = await request.json()
            processing_settings = body.get("processing_settings")
        except Exception:
            processing_settings = None
        result = await reframe_client.post(f"/reprocess/{photo_id}", json={"processing_settings": processing_settings})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_system_status():
    """Get system status information."""
    try:
        status = await reframe_client.get("/status")
        return status
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Hardware service unavailable: {str(e)}")

@app.post("/api/photos/{photo_id}/reprocess")
async def reprocess_single_photo(photo_id: str):
    """Reprocess a single photo to create missing dithered version."""
    try:
        result = await reframe_client.post(f"/reprocess/{photo_id}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reprocess photo: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
