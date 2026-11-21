from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus
import json
import feedparser
import requests
import logging
from plugins.base_plugin.base_plugin import BasePlugin


logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

CACHE_FILE = CACHE_DIR / "daily_summary.json"
CACHE_FILE_GOOGLE = CACHE_DIR / "google_output.json"

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"


# -------------------------
# HELPERS
# -------------------------

def _today():
    return datetime.now().strftime("%Y-%m-%d")


# -------------------------
# CACHE HELPERS
# -------------------------

def load_cache():
    """Retourne le résumé du jour (dict) ou None si absent, corrompu ou obsolète."""
    if not CACHE_FILE.exists():
        logger.debug("Pas de cache existant")
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
    except Exception as e:
        logger.error("Erreur de chargement du cache: %s", e)
        _delete_cache_file()
        return None

    summary = cache_data.get("summary")
    if cache_data.get("date") == _today() and isinstance(summary, dict):
        logger.debug("Cache valide pour %s", _today())
        return summary

    logger.debug("Cache obsolète ou invalide (date: %s)", cache_data.get("date"))
    return None


def save_cache(summary):
    """Sauvegarde le résumé avec la date du jour (encapsulation faite ici seulement)."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"date": _today(), "summary": summary}, f, ensure_ascii=False, indent=2)
        logger.debug("Cache sauvegardé pour %s", _today())
    except Exception as e:
        logger.error("Erreur de sauvegarde du cache: %s", e)


def _delete_cache_file():
    try:
        CACHE_FILE.unlink()
    except OSError:
        pass


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

        summary = load_cache()

        if summary is None:
            api_key = device_config.load_env_key("MISTRAL_API_KEY")
            if not api_key:
                raise RuntimeError("MISTRAL_API_KEY manquante")
            logger.debug("API Key chargée")

            word_count = self._to_int(settings.get("word_count"), 500)
            style = settings.get("style") or "journalistique"
            topics = settings.get("topics") or ""
            kids_filter = str(settings.get("kids_filter", False)).lower() in ("on", "true", "1", "yes")

            articles = self.fetch_google_news(topics)
            if not articles:
                raise RuntimeError("Aucun article récupéré depuis Google News. Vérifiez la connexion du Raspberry Pi.")
            logger.debug("Trouvé %d articles", len(articles))

            summary = self.call_mistral(api_key, articles, word_count, style, topics, kids_filter)
            # On ne met en cache qu'un résumé exploitable :
            # en cas d'échec Mistral, le prochain rafraîchissement réessaiera.
            if summary is not None:
                save_cache(summary)
            else:
                # Repli : utiliser un ancien cache plutôt qu'un écran vide
                summary = self._load_stale_cache()
                if summary is None:
                    raise RuntimeError("La synthèse Mistral a échoué. Vérifiez les logs du serveur.")

        else:
            logger.debug("Utilisation du cache du jour")

        summary = self._normalize_summary(summary)

        dimensions = device_config.get_resolution()
        if device_config.get_config("orientation") == "vertical":
            dimensions = dimensions[::-1]

        image = self.render_image(
            dimensions,
            html_file="digest.html",
            css_file="digest.css",
            template_params={
                "summary": summary,
                "date": datetime.now().strftime("%d/%m/%Y"),
                "plugin_settings": settings,
            },
        )

        if not image:
            raise RuntimeError("Failed to take screenshot, please check logs.")
        return image

    # -------------------------
    # NORMALISATION
    # -------------------------

    def _normalize_summary(self, summary):
        """Garantit une structure {theme: {titre: texte}} pour ce qui peut être parcouru en Jinja (texte brut, ancien format, dict imbriqués)."""
        normalized = {}
        if not isinstance(summary, dict):
            return normalized
        for theme, articles in summary.items():
            theme = str(theme)
            if isinstance(articles, dict):
                normalized[theme] = {
                    str(title): self._to_text(content)
                    for title, content in articles.items()
                }
            else:
                # Ancien format (texte brut sous un thème)
                normalized[theme] = {"Actualités": self._to_text(articles)}
        return normalized

    def _to_text(self, value):
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return " ".join(str(v) for v in value)
        if isinstance(value, dict):
            return " ".join("%s : %s" % (k, self._to_text(v)) for k, v in value.items())
        return str(value)

    def _to_int(self, value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _load_stale_cache(self):
        """Récupère le dernier cache connu, même périmé, en dépannage."""
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            summary = cache_data.get("summary")
            if isinstance(summary, dict):
                logger.warning("Utilisation d'un cache périmé (date: %s)", cache_data.get("date"))
                return summary
        except Exception:
            pass
        return None

    # -------------------------
    # GOOGLE NEWS RSS
    # -------------------------

    def fetch_google_news(self, topics):
        logger.debug("Fetching Google News")
        base_url = "https://news.google.com/rss/search?q="

        if topics:
            terms = [t.strip() for t in topics.split(",") if t.strip()]
            query = "+".join(quote_plus(t) for t in terms)
        else:
            query = quote_plus("actualité")

        url = "%s%s&hl=fr&gl=FR&ceid=FR:fr" % (base_url, query)
        logger.debug("URL Google News : %s", url)

        try:
            feed = feedparser.parse(url)
        except Exception as e:
            logger.error("Erreur de récupération du flux RSS : %s", e)
            return []

        articles = []
        for entry in feed.entries[:25]:
            source = entry.get("source", {})
            articles.append({
                "title": entry.get("title", ""),
                "source": source.get("title", "") if source else "",
                "date": entry.get("published", ""),
            })

        # Debug : sauvegarde d'une version allégée (titre + source) pour inspection
        try:
            with open(CACHE_FILE_GOOGLE, "w", encoding="utf-8") as f:
                json.dump({
                    "date": _today(),
                    "url": url,
                    "entries": articles,
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Erreur de sauvegarde du cache Google: %s", e)

        return articles

    # -------------------------
    # MISTRAL CALL
    # -------------------------

    def call_mistral(self, api_key, articles, word_count, style, topics, kids_filter):
        logger.debug("Appel de Mistral")

        formatted_news = "\n".join(
            "- %s (%s)" % (a["title"], a["source"])
            for a in articles
        )

        kid_instruction = ""
        if kids_filter:
            kid_instruction = (
                "Le contenu doit être adapté aux enfants, "
                "sans violence ni sujets anxiogènes. "
            )

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
    - Limite : 3 à 4 thèmes maximum, 2 à 3 actualités par thème.
    """

        logger.debug("Prompt envoyé à Mistral (%d caractères)", len(prompt))

        try:
            response = requests.post(
                MISTRAL_URL,
                headers={
                    "Authorization": "Bearer %s" % api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "model": "mistral-medium-latest",
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": "Tu es un journaliste professionnel. Réponds UNIQUEMENT en JSON valide, sans commentaire ni texte supplémentaire.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=120,
            )
            response.raise_for_status()
            raw_response = response.json()["choices"][0]["message"]["content"]

            # Extraction robuste du JSON (au cas où du texte entourerait l'objet)
            json_start = raw_response.find("{")
            json_end = raw_response.rfind("}") + 1
            if json_start < 0 or json_end <= json_start:
                logger.error("Réponse Mistral sans JSON : %s", raw_response[:500])
                return None

            parsed_json = json.loads(raw_response[json_start:json_end])
            logger.debug("Réponse Mistral parsée en JSON (%d thèmes)", len(parsed_json))
            return parsed_json

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Réponse Mistral non exploitable : %s", e)
            return None
        except requests.RequestException as e:
            logger.error("Erreur API Mistral : %s", e)
            return None
