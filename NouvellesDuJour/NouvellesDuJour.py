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
                "summary": summary,
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

        formatted_news = "\n".join(
            f"- {a['title']} ({a['source']})"
            for a in articles
        )

        kid_instruction = ""
        if kids_filter:
            kid_instruction = (
                "Le contenu doit être adapté aux enfants, "
                "sans violence ni sujets anxiogènes."
            )

        prompt = f"""
Tu es un rédacteur en chef d'un journal quotidien.

Voici les actualités du jour :

{formatted_news}

Consignes :
- Longueur : {word_count} mots
- Style : {style}
- Centres d'intérêt : {topics}

{kid_instruction}

Tu dois produire :
1. Un titre de une
2. Un résumé structuré des actualités importantes
3. Une synthèse claire et fluide

Le résultat doit être structurée de la façon:
news_item=[
    {"title": Titre 1, "summary": summary 1},
    {"title": Titre 2, "summary": summary 2},
.....
]

Ne mentionne pas la liste brute des articles.
"""
        logger.debug(f"prompt de requete vers Mistral: {prompt}")
        try:
            url = "https://api.mistral.ai/v1/chat/completions"
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "mistral-medium-latest",
                    "temperature": 0.7,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Tu es un journaliste professionnel."
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
            logger.debug(f"Mistral response: {response.json()["choices"][0]["message"]["content"]}")
            logger.debug(f"Mistral response 2: {response}")
            return response
            #return response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"Mistral API Error: {e}")
            return None
