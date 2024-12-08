import json
from pathlib import Path
import os

class Settings:
    def __init__(self):
        # Store settings in user's home directory
        self.settings_dir = Path.home() / ".git-cc"
        self.settings_file = self.settings_dir / "settings.json"
        self.settings = self._load_settings()

    def _load_settings(self):
        """Load settings from file or create default settings"""
        try:
            # Create config directory if it doesn't exist
            if not self.settings_dir.exists():
                os.makedirs(self.settings_dir, exist_ok=True)
            
            if not self.settings_file.exists():
                default_settings = {
                    "selected_model": "default"  # Changed from "mini" to "default"
                }
                self._save_settings(default_settings)
                return default_settings
            
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {"selected_model": "default"}  # Changed from "mini" to "default"
                
        except PermissionError:
            print(f"Warning: Cannot write to {self.settings_dir}. Using temporary settings.")
            return {"selected_model": "default"}  # Changed from "mini" to "default"

    def _save_settings(self, settings):
        """Save settings to file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except PermissionError:
            print(f"Warning: Cannot save settings to {self.settings_file}")

    def get_selected_model(self):
        """Get currently selected model"""
        model = self.settings.get("selected_model")
        return model if model else "default"  # Changed from "mini" to "default"

    def set_selected_model(self, model_name):
        """Set selected model"""
        self.settings["selected_model"] = model_name
        self._save_settings(self.settings)

    @staticmethod
    def get_config_path():
        """Get the path to the config directory"""
        return Path.home() / ".git-cc" 