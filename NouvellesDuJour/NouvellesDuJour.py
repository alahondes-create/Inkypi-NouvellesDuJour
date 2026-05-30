import json
import os
import requests
from pathlib import Path
from inkypi.plugins.base import BasePlugin

class NouvellesDuJour(BasePlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "News Synthesis"
        self.version = "1.0.0"
        self.config_file = Path(self.config_dir) / "news_synthesis_config.json"

    def get_config(self):
        """Charge la configuration depuis le fichier JSON."""
        if self.config_file.exists():
            with open(self.config_file, "r") as f:
                return json.load(f)
        return {
            "word_count": 200,
            "literary_style": "neutre",
            "interests": ["technologie", "économie"],
            "child_filter": False
        }

    def save_config(self, config):
        """Sauvegarde la configuration."""
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

    def fetch_news_synthesis(self, config):
        """Appelle l'API Mistral pour générer la synthèse."""
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Clé API Mistral non trouvée.")

        # Construction du prompt en fonction des réglages
        prompt = (
            f"Fais une synthèse de l'actualité du jour en {config['word_count']} mots, "
            f"dans le style de {config['literary_style']}. "
            f"Centres d'intérêt : {', '.join(config['interests'])}. "
            f"Filtre enfant : {'activé' if config['child_filter'] else 'désactivé'}."
        )

        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "mistral-tiny",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": config["word_count"] * 4  # Estimation
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Erreur API Mistral: {response.text}")

        return response.json()["choices"][0]["message"]["content"]

    def run(self):
        """Méthode principale appelée par InkyPi."""
        config = self.get_config()
        try:
            synthesis = self.fetch_news_synthesis(config)
            return synthesis
        except Exception as e:
            return f"Erreur: {str(e)}"

    def web_interface(self, app):
        """Route pour l'interface web de configuration."""
        from flask import request, render_template_string

        @app.route("/news_synthesis/config", methods=["GET", "POST"])
        def config_route():
            if request.method == "POST":
                new_config = {
                    "word_count": int(request.form.get("word_count", 200)),
                    "literary_style": request.form.get("literary_style", "neutre"),
                    "interests": [i.strip() for i in request.form.get("interests", "").split(",") if i.strip()],
                    "child_filter": request.form.get("child_filter") == "on"
                }
                self.save_config(new_config)
                return "Configuration sauvegardée !"
            else:
                config = self.get_config()
                return render_template_string("""
                <form method="POST">
                    <label>Nombre de mots: <input type="number" name="word_count" value="{{ config['word_count'] }}"></label><br>
                    <label>Style littéraire:
                        <select name="literary_style">
                            <option value="neutre" {% if config['literary_style'] == 'neutre' %}selected{% endif %}>Neutre</option>
                            <option value="Victor Hugo" {% if config['literary_style'] == 'Victor Hugo' %}selected{% endif %}>Victor Hugo</option>
                            <option value="Flaubert" {% if config['literary_style'] == 'Flaubert' %}selected{% endif %}>Flaubert</option>
                        </select>
                    </label><br>
                    <label>Centres d'intérêt (séparés par des virgules): <input type="text" name="interests" value="{{ ', '.join(config['interests']) }}"></label><br>
                    <label><input type="checkbox" name="child_filter" {% if config['child_filter'] %}checked{% endif %}> Filtre enfant</label><br>
                    <button type="submit">Sauvegarder</button>
                </form>
                """, config=config)

# Instanciation du plugin
plugin = NewsSynthesisPlugin()