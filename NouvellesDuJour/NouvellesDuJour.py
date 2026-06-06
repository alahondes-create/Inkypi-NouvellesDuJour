from datetime import datetime
from pathlib import Path
import json
import feedparser
import requests
import logging
from plugins.base_plugin.base_plugin import BasePlugin
from collections import defaultdict


logger = logging.getLogger(__name__)
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

CACHE_FILE = CACHE_DIR / "daily_summary.json"
CACHE_FILE_GOOGLE= CACHE_DIR / "google_output.json"

# -------------------------
# CACHE HELPERS
# -------------------------

def load_cache():
    if CACHE_FILE.exists():
        try:
            logger.debug("Loading cache")
            return json.load(open(CACHE_FILE, "r", encoding="utf-8"))
        except:
            logger.debug("No cache")
            return None
    return None


def save_cache(data):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        logger.debug("Saving cache")
        json.dump(data, f, ensure_ascii=False, indent=2)


# -------------------------
# MAIN PLUGIN
# -------------------------

class NouvellesDuJour(BasePlugin):

    def generate_settings_template(self):
        logger.debug("Generating settings template")
        template = super().generate_settings_template()
        template["style_settings"] = True
        return template

    def generate_image(self, settings, device_config):
        logger.debug("Generating image")
        api_key = device_config.load_env_key("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY manquante")
        logger.debug("API Key loaded")
        word_count = int(settings.get("word_count", 500))
        style = settings.get("style", "journalistique")
        topics = settings.get("topics", "")
        kids_filter = settings.get("kids_filter", False)

        articles = self.fetch_google_news(topics)
        logger.debug(f"Found {len(articles)} articles")
        summary = self.get_or_create_summary(
             api_key,
             articles,
             word_count,
             style,
             topics,
             kids_filter
         )

        dimensions = device_config.get_resolution()
        if device_config.get_config("orientation") == "vertical":
            dimensions = dimensions[::-1]

        #template_params["plugin_settings"] = settings
        #summary="toto"
        image= self.render_image(
            dimensions,
            html_file="digest.html",
            css_file="digest.css",
            template_params={
                "summary_mistral": summary,
                "date": datetime.now().strftime("%d/%m/%Y"),
                "plugin_settings": settings,

            }
        )


        if not image:
            raise RuntimeError("Failed to take screenshot,please check logs.")
        return image

    # -------------------------
    # GOOGLE NEWS RSS
    # -------------------------

    def fetch_google_news(self, topics):
        logger.debug("Fetching Google News")
        base_url = "https://news.google.com/rss/search?q="

        if topics:
            query = "+".join(topics.split(","))
        else:
            query = "actualité"

        url = f"{base_url}{query}&hl=fr&gl=FR&ceid=FR:fr"

        feed = feedparser.parse(url)
        with open(CACHE_FILE_GOOGLE, "w", encoding="utf-8") as f:
            logger.debug("Saving cache google news")
            json.dump(feed, f, ensure_ascii=False, indent=2)

        articles = []

        for entry in feed.entries[:25]:

            articles.append({
                "title": entry.title,
                "source": getattr(entry, "source", {}).get("title", ""),
                "date": getattr(entry, "published", ""),
            })

        return articles

    # -------------------------
    # CACHE LOGIC
    # -------------------------

    def get_or_create_summary(
        self,
        api_key,
        articles,
        word_count,
        style,
        topics,
        kids_filter
    ):

        today = datetime.now().strftime("%Y-%m-%d")

        cache = load_cache()

        if cache and cache.get("date") == today:
            logger.debug(f"Lecture du cache d aujourd'hui : {cache.get("date")}")
            logger.debug(f"Sommaire de l'actualité d'aujourdhui: {cache["summary"]}")
            return cache["summary"]

        summary = self.call_mistral(
            api_key,
            articles,
            word_count,
            style,
            topics,
            kids_filter
        )

        save_cache({
            "date": today,
            "summary": summary
        })

        return summary

    # -------------------------
    # MISTRAL CALL
    # -------------------------

    def call_mistral(
            self,
            api_key,
            articles,
            word_count,
            style,
            topics,
            kids_filter
    ):
        logger.debug("Calling Mistral")

        # Formatage des articles pour le prompt
        formatted_news = "\n".join(
            f"- {a['title']} ({a['source']}) : {a.get('description', '')}"
            for a in articles
        )

        # Instruction pour le filtre enfant
        kid_instruction = ""
        if kids_filter:
            kid_instruction = (
                "Le contenu doit être adapté aux enfants, "
                "sans violence ni sujets anxiogènes. "
            )

        # Prompt strict pour un format JSON
        prompt = f"""
    Tu es un rédacteur en chef. Voici les actualités du jour :

    {formatted_news}

    **Consignes strictes :**
    - Longueur totale : environ {word_count} mots.
    - Style : {style}.
    - Centres d'intérêt prioritaires : {topics}.
    {kid_instruction}

    **Format de sortie obligatoire (JSON) :**
    Réponds UNIQUEMENT avec un objet JSON structuré comme ceci :
    {{
      "Centre d'intérêt 1": {{
        "Titre de l'actualité 1": "Résumé concis de l'actualité 1",
        "Titre de l'actualité 2": "Résumé concis de l'actualité 2"
      }},
      "Centre d'intérêt 2": {{
        "Titre de l'actualité 3": "Résumé concis de l'actualité 3"
      }}
    }}

    **Règles :**
    - Ne pas inclure de texte en dehors du JSON.
    - Les clés doivent être les centres d'intérêt ou les titres des actualités.
    - Les valeurs doivent être des résumés en français, clairs et fluides.
    - Si un centre d'intérêt n'a pas d'actualité, ne l'inclus pas.
    """

        logger.debug(f"Prompt envoyé à Mistral : {prompt}")

        try:
            url = "https://api.mistral.ai/v1/chat/completions"
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "mistral-medium-latest",
                    "temperature": 0.3,  # Temp plus basse pour un format strict
                    "messages": [
                        {
                            "role": "system",
                            "content": "Tu es un journaliste professionnel. Réponds UNIQUEMENT en JSON valide, sans commentaire ni texte supplémentaire."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                timeout=120
            )

            response.raise_for_status()
            raw_response = response.json()["choices"][0]["message"]["content"]

            # Vérification que la réponse est du JSON valide
            try:
                import json
                # On nettoie la réponse pour extraire le JSON (au cas où Mistral ajouterait du texte)
                json_start = raw_response.find("{")
                json_end = raw_response.rfind("}") + 1
                json_str = raw_response[json_start:json_end]
                parsed_json = json.loads(json_str)
                logger.debug(f"Réponse Mistral (JSON) : {parsed_json}")
                return parsed_json
            except json.JSONDecodeError as e:
                logger.error(f"La réponse de Mistral n'est pas du JSON valide : {e}\nRéponse brute : {raw_response}")
                # Retourne un JSON par défaut en cas d'erreur
                return {
                    "Erreur": {
                        "Format invalide": "La réponse de Mistral n'a pas pu être parsée en JSON. Vérifiez le prompt ou la réponse brute."
                    }
                }

        except Exception as e:
            logger.error(f"Erreur API Mistral : {e}")
            return None
