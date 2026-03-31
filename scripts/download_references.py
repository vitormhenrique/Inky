#!/usr/bin/env python3
"""Download public-domain reference paintings from Wikimedia Commons.

All works listed here are pre-1900 and in the public domain worldwide.
Uses the Wikimedia Commons API to resolve proper thumbnail URLs at allowed
sizes, avoiding the 403/429 errors from direct thumbnail URL construction.

Usage:
    uv run python scripts/download_references.py
    uv run python scripts/download_references.py --style renaissance_portrait
    uv run python scripts/download_references.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# ── Project root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STYLES_DIR = PROJECT_ROOT / "data" / "styles"

# Polite delay between downloads (seconds)
DOWNLOAD_DELAY = 1.0

# Target width for thumbnails (must be an allowed Wikimedia step)
THUMB_WIDTH = 1024

# User-Agent for Wikimedia API (required by their policy)
USER_AGENT = (
    "InkyStylize/0.1 (https://github.com/inky-stylize; "
    "image-stylization-project) Python/3 urllib"
)

# Wikimedia Commons API endpoint
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# ── Reference painting catalogue ─────────────────────────────────────────────
# Each entry: (local_filename, wikimedia_commons_filename)
# The commons filename is the "File:..." title on Wikimedia Commons.
# The API resolves it to a proper download URL.

CATALOGUE: dict[str, list[tuple[str, str]]] = {
    # ═════════════════════════════════════════════════════════════════════════
    # RENAISSANCE PORTRAIT
    # ═════════════════════════════════════════════════════════════════════════
    "renaissance_portrait": [
        (
            "leonardo_mona_lisa.jpg",
            "Mona Lisa, by Leonardo da Vinci, from C2RMF retouched.jpg",
        ),
        (
            "leonardo_lady_ermine.jpg",
            "Lady with an Ermine - Leonardo da Vinci (adjusted levels).jpg",
        ),
        ("raphael_woman_veil.jpg", "La velada, por Rafael.jpg"),
        (
            "botticelli_birth_venus.jpg",
            "Sandro Botticelli - La nascita di Venere - Google Art Project - edited.jpg",
        ),
        ("raphael_self_portrait.jpg", "Raffaello Sanzio.jpg"),
        (
            "titian_man_blue_sleeve.jpg",
            "Titian - Portrait of a man with a quilted sleeve.jpg",
        ),
        (
            "bellini_doge_loredan.jpg",
            "Giovanni Bellini, portrait of Doge Leonardo Loredan.jpg",
        ),
        (
            "leonardo_ginevra.jpg",
            "Leonardo da Vinci - Ginevra de' Benci - Google Art Project.jpg",
        ),
        (
            "botticelli_venus_mars.jpg",
            "Sandro Botticelli - Venus and Mars - WGA2776.jpg",
        ),
        (
            "raphael_baldassare.jpg",
            "Baldassare Castiglione, by Raffaello Sanzio, from C2RMF retouched.jpg",
        ),
    ],
    # ═════════════════════════════════════════════════════════════════════════
    # BAROQUE OIL PAINTING
    # ═════════════════════════════════════════════════════════════════════════
    "baroque_oil_painting": [
        ("caravaggio_judith.jpg", "Caravaggio Judith Beheading Holofernes.jpg"),
        ("caravaggio_bacchus.jpg", "Bacchus-Caravaggio (1595).jpg"),
        (
            "velazquez_las_meninas.jpg",
            "Las Meninas, by Diego Velázquez, from Prado in Google Earth.jpg",
        ),
        (
            "velazquez_prince_balthasar.jpg",
            "Diego Velázquez - Prince Baltasar Carlos as a Hunter - WGA24384.jpg",
        ),
        (
            "rubens_self_portrait.jpg",
            "Peter Paul Rubens - Self-Portrait - WGA20380.jpg",
        ),
        (
            "artemisia_self_portrait.jpg",
            "Self-portrait as the Allegory of Painting (La Pittura) - Artemisia Gentileschi.jpg",
        ),
        ("caravaggio_narcissus.jpg", "Narcissus-Caravaggio (1594-96) edited.jpg"),
        ("vermeer_girl_pearl_earring.jpg", "Meisje met de parel.jpg"),
        (
            "rubens_samson_delilah.jpg",
            "Peter Paul Rubens - Samson and Delilah - Google Art Project.jpg",
        ),
        (
            "velazquez_bufon.jpg",
            "Velázquez – Bufón don Sebastián de Morra (Museo del Prado, c. 1645).jpg",
        ),
    ],
    # ═════════════════════════════════════════════════════════════════════════
    # ROCOCO PORTRAIT
    # ═════════════════════════════════════════════════════════════════════════
    "rococo_portrait": [
        ("fragonard_swing.jpg", "Fragonard, The Swing.jpg"),
        ("fragonard_reader.jpg", "Fragonard, The Reader.jpg"),
        ("boucher_madame_pompadour.jpg", "François Boucher - Madame de Pompadour.jpg"),
        (
            "vigee_lebrun_self_portrait.jpg",
            "Self-portrait in a Straw Hat by Elisabeth-Louise Vigée-Lebrun.jpg",
        ),
        ("vigee_lebrun_marie_antoinette.jpg", "Marie Antoinette Adult.jpg"),
        ("watteau_pierrot.jpg", "Jean-Antoine Watteau - Pierrot.jpg"),
        (
            "boucher_diana_bath.jpg",
            "François Boucher - Diana Leaving Her Bath - WGA02901.jpg",
        ),
        (
            "watteau_embarkation.jpg",
            "L'Embarquement pour Cythere, by Antoine Watteau, from C2RMF retouched.jpg",
        ),
        (
            "gainsborough_blue_boy.jpg",
            "Thomas Gainsborough - The Blue Boy (The Huntington Library, San Marino L. A.).jpg",
        ),
        (
            "fragonard_music_lesson.jpg",
            "Jean-Honoré Fragonard - The Music Lesson (1769).jpg",
        ),
    ],
    # ═════════════════════════════════════════════════════════════════════════
    # DUTCH GOLDEN AGE
    # ═════════════════════════════════════════════════════════════════════════
    "dutch_golden_age": [
        (
            "rembrandt_self_portrait_1659.jpg",
            "Rembrandt van Rijn - Self-Portrait - Google Art Project.jpg",
        ),
        ("rembrandt_night_watch.jpg", "The Night Watch - HD.jpg"),
        (
            "vermeer_milkmaid.jpg",
            "Johannes Vermeer - Het melkmeisje - Google Art Project.jpg",
        ),
        ("vermeer_girl_pearl.jpg", "Meisje met de parel.jpg"),
        (
            "hals_laughing_cavalier.jpg",
            "Frans Hals - Portret van een man (de 'Lachende Cavalier') - WGA11116.jpg",
        ),
        (
            "rembrandt_anatomy_lesson.jpg",
            "Rembrandt - The Anatomy Lesson of Dr Nicolaes Tulp.jpg",
        ),
        (
            "rembrandt_storm.jpg",
            "Rembrandt Christ in the Storm on the Lake of Galilee.jpg",
        ),
        (
            "vermeer_art_painting.jpg",
            "Johannes Vermeer - The Art of Painting - Google Art Project.jpg",
        ),
        (
            "rembrandt_bathsheba.jpg",
            "Rembrandt - Bathsheba at Her Bath (Metropolitan Museum of Art).jpg",
        ),
        ("hals_malle_babbe.jpg", "Frans Hals - Malle Babbe - WGA11098.jpg"),
    ],
    # ═════════════════════════════════════════════════════════════════════════
    # ROMANTICISM
    # ═════════════════════════════════════════════════════════════════════════
    "romanticism": [
        (
            "delacroix_liberty.jpg",
            "Eugène Delacroix - Le 28 Juillet. La Liberté guidant le peuple.jpg",
        ),
        (
            "turner_fighting_temeraire.jpg",
            "The Fighting Temeraire, JMW Turner, National Gallery.jpg",
        ),
        (
            "friedrich_wanderer_above_sea.jpg",
            "Caspar David Friedrich - Wanderer above the sea of fog.jpg",
        ),
        (
            "turner_rain_steam_speed.jpg",
            "Turner - Rain, Steam and Speed - National Gallery file.jpg",
        ),
        (
            "gericault_raft_medusa.jpg",
            "JEAN LOUIS THÉODORE GÉRICAULT - La Balsa de la Medusa (Museo del Louvre, 1818-19).jpg",
        ),
        (
            "delacroix_death_sardanapalus.jpg",
            "Eugène Delacroix - La Mort de Sardanapale.jpg",
        ),
        ("turner_slave_ship.jpg", "Slave-ship.jpg"),
        (
            "friedrich_abbey_oak.jpg",
            "Caspar David Friedrich - Abtei im Eichwald - Google Art Project.jpg",
        ),
        (
            "cole_oxbow.jpg",
            "Cole Thomas The Oxbow (The Connecticut River near Northampton 1836).jpg",
        ),
        (
            "bierstadt_rocky_mountains.jpg",
            "Albert Bierstadt - The Rocky Mountains, Lander's Peak.jpg",
        ),
    ],
    # ═════════════════════════════════════════════════════════════════════════
    # IMPRESSIONISM
    # ═════════════════════════════════════════════════════════════════════════
    "impressionism": [
        ("monet_impression_sunrise.jpg", "Monet - Impression, Sunrise.jpg"),
        (
            "monet_water_lilies_1906.jpg",
            "Claude Monet - Water Lilies - 1906, Ryerson.jpg",
        ),
        (
            "renoir_moulin_galette.jpg",
            "Pierre-Auguste Renoir, Le Moulin de la Galette.jpg",
        ),
        (
            "renoir_boating_party.jpg",
            "Pierre-Auguste Renoir - Luncheon of the Boating Party - Google Art Project.jpg",
        ),
        (
            "degas_ballet_class.jpg",
            "Edgar Degas - The Ballet Class - Google Art Project.jpg",
        ),
        (
            "monet_haystacks_sunset.jpg",
            "Claude Monet - Stacks of Wheat (End of Summer) - 1985.1103 - Art Institute of Chicago.jpg",
        ),
        ("morisot_cradle.jpg", "Berthe Morisot - The Cradle - Google Art Project.jpg"),
        (
            "cassatt_childs_bath.jpg",
            "Mary Cassatt - The Child's Bath - Google Art Project.jpg",
        ),
        (
            "monet_rouen_cathedral.jpg",
            "Claude Monet - Rouen Cathedral, Facade (Sunset).jpg",
        ),
        (
            "monet_japanese_bridge.jpg",
            "Claude Monet - The Japanese Footbridge - Google Art Project.jpg",
        ),
        (
            "renoir_girl_watering_can.jpg",
            "Auguste Renoir - A Girl with a Watering Can - Google Art Project.jpg",
        ),
    ],
    # ═════════════════════════════════════════════════════════════════════════
    # POST-IMPRESSIONISM
    # ═════════════════════════════════════════════════════════════════════════
    "post_impressionism": [
        (
            "vangogh_starry_night.jpg",
            "Van Gogh - Starry Night - Google Art Project.jpg",
        ),
        (
            "vangogh_self_portrait_1889.jpg",
            "Vincent van Gogh - Self-Portrait - Google Art Project.jpg",
        ),
        (
            "vangogh_bedroom.jpg",
            "Vincent van Gogh - De slaapkamer - Google Art Project.jpg",
        ),
        (
            "cezanne_mont_sainte_victoire.jpg",
            "Paul Cézanne - La Montagne Sainte-Victoire vue du bosquet du Château Noir.jpg",
        ),
        (
            "cezanne_card_players.jpg",
            "Les Joueurs de cartes, par Paul Cézanne, Yorck.jpg",
        ),
        (
            "gauguin_spirit_dead.jpg",
            "Paul Gauguin - Spirit of the Dead Watching - Google Art Project.jpg",
        ),
        (
            "seurat_grande_jatte.jpg",
            "A Sunday on La Grande Jatte, Georges Seurat, 1884.jpg",
        ),
        (
            "vangogh_cafe_night.jpg",
            "Van Gogh - Terrasse des Cafés an der Place du Forum in Arles am Abend1.jpeg",
        ),
        ("vangogh_sunflowers.jpg", "Vincent Willem van Gogh 127.jpg"),
        (
            "toulouse_lautrec_moulin_rouge.jpg",
            "Henri de Toulouse-Lautrec - At the Moulin Rouge - Google Art Project.jpg",
        ),
        (
            "gauguin_tahitian_women.jpg",
            "Paul Gauguin - Tahitian Women on the Beach - Google Art Project.jpg",
        ),
    ],
    # ═════════════════════════════════════════════════════════════════════════
    # VICTORIAN ANIMAL PORTRAIT
    # ═════════════════════════════════════════════════════════════════════════
    "victorian_animal_portrait": [
        (
            "landseer_monarch_glen.jpg",
            "Edwin Landseer - The Monarch of the Glen - Google Art Project.jpg",
        ),
        (
            "landseer_dignity_impudence.jpg",
            "Edwin Henry Landseer - Dignity and Impudence.jpg",
        ),
        (
            "landseer_newfoundland.jpg",
            "A Distinguished Member of the Humane Society by Sir Edwin Landseer.jpg",
        ),
        ("stubbs_whistlejacket.jpg", "Whistlejacket by George Stubbs.jpg"),
        ("landseer_old_shepherds.jpg", "Landseer The Old Shepherd's Chief Mourner.jpg"),
        (
            "rosa_bonheur_horse_fair.jpg",
            "Rosa Bonheur - The Horse Fair - Google Art Project.jpg",
        ),
        ("landseer_lions_study.jpg", "Lion-landseer-study.jpg"),
        (
            "agasse_nubian_giraffe.jpg",
            "Jacques-Laurent Agasse - The Nubian Giraffe.jpg",
        ),
        ("barker_girl_with_dogs.jpg", "Charles Burton Barber - Girl with Dogs.jpg"),
        ("barber_suspense.jpg", "Suspense-Barber.jpg"),
        (
            "landseer_alpine_mastiffs.jpg",
            "Edwin Landseer - Alpine Mastiffs Reanimating a Distressed Traveller - Google Art Project.jpg",
        ),
    ],
    # ═════════════════════════════════════════════════════════════════════════
    # CLASSICAL EQUESTRIAN
    # ═════════════════════════════════════════════════════════════════════════
    "classical_equestrian": [
        ("stubbs_whistlejacket.jpg", "Whistlejacket by George Stubbs.jpg"),
        (
            "stubbs_mares_foals.jpg",
            "George Stubbs - Mares and Foals in a Landscape - Google Art Project.jpg",
        ),
        (
            "david_napoleon_alps.jpg",
            "David - Napoleon crossing the Alps - Malmaison2.jpg",
        ),
        (
            "velazquez_prince_horse.jpg",
            "Retrato ecuestre del Príncipe Baltasar Carlos (Velázquez).jpg",
        ),
        (
            "rubens_duke_lerma.jpg",
            "Rubens - Equestrian Portrait of the Duke of Lerma - Prado.jpg",
        ),
        (
            "delacroix_fantasia.jpg",
            "Eugène Delacroix - Fantasia Arabe - Google Art Project.jpg",
        ),
        (
            "rosa_bonheur_horse_fair.jpg",
            "Rosa Bonheur - The Horse Fair - Google Art Project.jpg",
        ),
        (
            "stubbs_horse_lion.jpg",
            "George Stubbs - Horse Attacked by a Lion (Tate Britain).jpg",
        ),
        ("gericault_horse_race.jpg", "Jean Louis Théodore Géricault 005.jpg"),
        ("stubbs_hambletonian.jpg", "George Stubbs Hambletonian.jpg"),
    ],
    # ═════════════════════════════════════════════════════════════════════════
    # NATURALIST OIL PORTRAIT
    # ═════════════════════════════════════════════════════════════════════════
    "naturalist_oil_portrait": [
        (
            "sargent_madame_x.jpg",
            "Madame X (Madame Pierre Gautreau), John Singer Sargent, 1884 (unfree frame crop).jpg",
        ),
        (
            "sargent_carnation_lily.jpg",
            "John Singer Sargent - Carnation, Lily, Lily, Rose - Google Art Project.jpg",
        ),
        (
            "sargent_daughters_boit.jpg",
            "John Singer Sargent - The Daughters of Edward Darley Boit 1882.jpg",
        ),
        ("whistler_mother.jpg", "Whistlers Mother high res.jpg"),
        (
            "eakins_gross_clinic.jpg",
            "Thomas Eakins, American - Portrait of Dr. Samuel D. Gross (The Gross Clinic) - Google Art Project.jpg",
        ),
        ("sargent_lady_agnew.jpg", "Sargent - Lady Agnew.jpg"),
        (
            "chase_seaside.jpg",
            "William Merritt Chase - At the Seaside - Google Art Project.jpg",
        ),
        ("sorolla_walk_beach.jpg", "Paseo a orillas del mar.jpg"),
        ("zorn_self_portrait.jpg", "Anders Zorn - Självporträtt i rött.jpg"),
        ("sargent_el_jaleo.jpg", "El Jaleo by John Singer Sargent 1882.jpg"),
    ],
    # ═════════════════════════════════════════════════════════════════════════
    # CUBISM — angular forms, fractured planes, geometric abstraction
    # Great for NST: strong geometric patterns transfer well
    # ═════════════════════════════════════════════════════════════════════════
    "cubism": [
        (
            "gris_portrait_picasso.jpg",
            "Juan Gris - Portrait of Pablo Picasso - Google Art Project.jpg",
        ),
        (
            "gris_checkered_tablecloth.jpg",
            "Still Life with Checked Tablecloth Juan Gris 1915.jpeg",
        ),
        (
            "gris_violin_glass.jpg",
            "Juan Gris - Violin and Glass - 1963.117 - Fogg Museum.jpg",
        ),
        (
            "leger_nudes_forest.jpg",
            "Fernand Léger, 1910, Nudes in the forest (Nus dans la forêt), oil on canvas, 120 x 170 cm, Kröller-Müller Museum.jpg",
        ),
        (
            "leger_contraste_formes.jpg",
            "Contraste de formes, 1913 - Fernand Léger.jpg",
        ),
        (
            "delaunay_simultaneous_windows.jpg",
            "Robert Delaunay, 1912, Les Fenêtres simultanée sur la ville (Simultaneous Windows on the City), 40 x 46 cm, Kunsthalle Hamburg.jpg",
        ),
        (
            "delaunay_windows_open.jpg",
            "'Windows Open Simultaneously (First Part, Third Motif)' by Robert Delaunay.JPG",
        ),
        (
            "delaunay_disque_simultane.jpg",
            "Delaunay Disque simultané.jpg",
        ),
        (
            "gleizes_bathers.jpg",
            "Albert Gleizes, 1912, Les Baigneuses, oil on canvas, 105 x 171 cm, Paris, Musée d'Art Moderne de la Ville de Paris.jpg",
        ),
        (
            "picabia_udnie.jpg",
            "Francis picabia, udnie, 1913, 01.jpg",
        ),
        (
            "popova_painterly_architectonic.jpg",
            "Lyubov Popova - Painterly Architectonic - GMA 2080 - National Galleries of Scotland.jpg",
        ),
        (
            "mondrian_composition_rby.jpg",
            "Piet Mondriaan, 1930 - Mondrian Composition II in Red, Blue, and Yellow.jpg",
        ),
        (
            "malevich_suprematism.jpg",
            "Suprematism by Malevich (1915, GRM).jpg",
        ),
        (
            "malevich_supremus55.jpg",
            "Supremus 55 (Malevich, 1916).jpg",
        ),
    ],
    # ═════════════════════════════════════════════════════════════════════════
    # FAUVISM & BOLD COLOR — vivid, expressive, wild brushwork
    # Ideal for NST: strong saturated colors produce stunning transfers
    # ═════════════════════════════════════════════════════════════════════════
    "fauvism_bold_color": [
        (
            "matisse_woman_hat.jpg",
            "Matisse-Woman-with-a-Hat.jpg",
        ),
        (
            "matisse_harmony_red.jpg",
            "Matisse-The-Dessert-Harmony-in-Red-Henri-1908-fast.jpg",
        ),
        (
            "matisse_dance.jpg",
            "La Danse II, par Henri Matisse.jpg",
        ),
        (
            "matisse_blue_nude.jpg",
            "Nu bleu (souvenir de Biskra), par Henri Matisse.jpg",
        ),
        (
            "derain_charing_cross.jpg",
            "Derain CharingCrossBridge.png",
        ),
        (
            "derain_turning_road.jpg",
            "The Turning Road, L´Estaque.jpg",
        ),
        (
            "derain_mountains_collioure.jpg",
            "Montagnes à Collioure, par André Derain.jpg",
        ),
        (
            "marc_blue_horse.jpg",
            "Marc, Franz - Blue Horse I - Google Art Project.jpg",
        ),
        (
            "marc_fate_animals.jpg",
            "Franz Marc-The fate of the animals-1913.jpg",
        ),
        (
            "macke_promenade.jpg",
            "August Macke - Promenade - G 13328 - Lenbachhaus.jpg",
        ),
        (
            "macke_garden_lake_thun.jpg",
            "August Macke - Garden on Lake Thun (Pomegranate Tree and Palm in the Garden), 1914 - Google Art Project.jpg",
        ),
        (
            "kandinsky_composition_vii.jpg",
            "Vassily Kandinsky, 1913 - Composition 7.jpg",
        ),
        (
            "kandinsky_yellow_red_blue.jpg",
            "Kandinsky - Jaune Rouge Bleu.jpg",
        ),
        (
            "kandinsky_several_circles.jpg",
            "Vassily Kandinsky, 1926 - Several Circles, Gugg 0910 25.jpg",
        ),
        (
            "klee_senecio.jpg",
            "Senecio2.JPG",
        ),
    ],
    # ═════════════════════════════════════════════════════════════════════════
    # EXPRESSIONISM — bold distortion, emotional intensity, vivid palette
    # Excellent for NST: dramatic contrasts and bold forms transfer powerfully
    # ═════════════════════════════════════════════════════════════════════════
    "expressionism_bold_color": [
        (
            "munch_scream.jpg",
            "Edvard Munch, 1893, The Scream, oil, tempera and pastel on cardboard, 91 x 73 cm, National Gallery of Norway.jpg",
        ),
        (
            "munch_madonna.jpg",
            "Edvard Munch - Madonna (1894-1895).jpg",
        ),
        (
            "kirchner_street_berlin.jpg",
            "Ernst Ludwig Kirchner, 1913, Street, Berlin, oil on canvas, 120.6 x 91.1 cm, MoMA.jpg",
        ),
        (
            "kirchner_marcella.jpg",
            "Ernst Ludwig Kirchner - Artistin (Marzella).jpg",
        ),
        (
            "schiele_self_portrait.jpg",
            "Egon Schiele - Self-Portrait with Physalis - Google Art Project.jpg",
        ),
        (
            "klimt_kiss.jpg",
            "The Kiss - Gustav Klimt - Google Cultural Institute.jpg",
        ),
        (
            "klimt_adele_bloch_bauer.jpg",
            "Gustav Klimt, 1907, Adele Bloch-Bauer I, Neue Galerie New York.jpg",
        ),
        (
            "klimt_tree_of_life.jpg",
            "Gustav Klimt 032.jpg",
        ),
        (
            "jawlensky_schokko.jpg",
            "Alexej von Jawlensky - Schokko mit Tellerhut.jpg",
        ),
        (
            "jawlensky_schokko_red_hat.jpg",
            "Alexej Jawlensky - Schokko with Red Hat (1909).jpg",
        ),
        (
            "macke_lady_green_jacket.jpg",
            "August Macke 005.jpg",
        ),
    ],
}


# ── Wikimedia Commons API helpers ────────────────────────────────────────────


def _api_request(params: dict[str, str]) -> dict:
    """Make a Wikimedia Commons API request and return parsed JSON."""
    params["format"] = "json"
    url = f"{COMMONS_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def resolve_image_urls(
    commons_filenames: list[str], thumb_width: int = THUMB_WIDTH
) -> dict[str, str]:
    """Resolve Commons filenames to download URLs via the API.

    Processes in batches of 50 (API limit).
    Returns {commons_filename: download_url}.
    """
    result: dict[str, str] = {}

    # Batch into groups of 50
    titles = [f"File:{name}" for name in commons_filenames]
    for i in range(0, len(titles), 50):
        batch = titles[i : i + 50]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": str(thumb_width),
        }
        try:
            data = _api_request(params)
        except (urllib.error.URLError, OSError) as e:
            print(f"  API error: {e}")
            continue

        pages = data.get("query", {}).get("pages", {})
        for _pid, page in pages.items():
            title = page.get("title", "")
            # Strip "File:" prefix to get back the commons filename
            commons_name = title.removeprefix("File:")
            imageinfo = page.get("imageinfo", [])
            if imageinfo:
                # Prefer thumburl (resized), fall back to source url
                url = imageinfo[0].get("thumburl") or imageinfo[0].get("url", "")
                if url:
                    result[commons_name] = url

        if i + 50 < len(titles):
            time.sleep(0.5)  # polite delay between API batches

    return result


# ── Download logic ───────────────────────────────────────────────────────────


def _download_file(url: str, dest: Path) -> bool:
    """Download a single file. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 1024:
        return True  # already downloaded

    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            if len(data) < 1024:
                print(
                    f"  WARN: tiny response for {dest.name} ({len(data)} bytes) — skipping"
                )
                return False
            dest.write_bytes(data)
            return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        print(f"  FAIL: {dest.name} — {e}")
        return False


def download_style(style_name: str) -> tuple[int, int]:
    """Download all references for one style. Returns (success, total)."""
    entries = CATALOGUE.get(style_name, [])
    if not entries:
        print(f"  No entries for '{style_name}'")
        return 0, 0

    style_dir = STYLES_DIR / style_name

    # Check which files we actually need to download
    needed: list[tuple[str, str]] = []
    already = 0
    for local_name, commons_name in entries:
        dest = style_dir / local_name
        if dest.exists() and dest.stat().st_size > 1024:
            already += 1
        else:
            needed.append((local_name, commons_name))

    if already:
        print(f"  {already} already downloaded, {len(needed)} remaining")

    if not needed:
        return len(entries), len(entries)

    # Resolve URLs via API
    commons_names = [cn for _, cn in needed]
    print(f"  Resolving {len(commons_names)} URLs via Wikimedia API…")
    url_map = resolve_image_urls(commons_names)

    ok = already
    for local_name, commons_name in needed:
        url = url_map.get(commons_name)
        if not url:
            print(f"  SKIP: {local_name} — could not resolve URL for '{commons_name}'")
            continue
        dest = style_dir / local_name
        if _download_file(url, dest):
            ok += 1
            size_kb = dest.stat().st_size // 1024
            print(f"  ✓ {local_name} ({size_kb} KB)")
        time.sleep(DOWNLOAD_DELAY)

    return ok, len(entries)


def download_all(only_style: str | None = None) -> None:
    """Download reference paintings for all (or one) style(s)."""
    styles = [only_style] if only_style else list(CATALOGUE.keys())
    grand_ok = 0
    grand_total = 0

    for style in styles:
        print(f"\n{'─' * 50}")
        print(f"  {style}")
        print(f"{'─' * 50}")
        ok, total = download_style(style)
        grand_ok += ok
        grand_total += total
        print(f"  {ok}/{total} downloaded")

    print(f"\n{'═' * 50}")
    print(f"  TOTAL: {grand_ok}/{grand_total} reference paintings")
    print(f"  Location: {STYLES_DIR}")
    print(f"{'═' * 50}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download public-domain reference paintings from Wikimedia Commons."
    )
    parser.add_argument(
        "--style",
        type=str,
        default=None,
        help="Download only this style (e.g. 'impressionism'). Default: all styles.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List paintings without downloading.",
    )
    parser.add_argument(
        "--list-styles",
        action="store_true",
        help="Print available style names and exit.",
    )
    args = parser.parse_args()

    if args.list_styles:
        for name, entries in CATALOGUE.items():
            print(f"  {name:<30s}  ({len(entries)} paintings)")
        total = sum(len(e) for e in CATALOGUE.values())
        print(f"\n  Total: {total} paintings across {len(CATALOGUE)} styles")
        sys.exit(0)

    if args.dry_run:
        for name, entries in CATALOGUE.items():
            if args.style and name != args.style:
                continue
            print(f"\n{name}/")
            for local_name, commons_name in entries:
                dest = STYLES_DIR / name / local_name
                status = (
                    "EXISTS"
                    if (dest.exists() and dest.stat().st_size > 1024)
                    else "MISSING"
                )
                print(f"  [{status}] {local_name}  ←  {commons_name}")
        sys.exit(0)

    print("Downloading reference paintings from Wikimedia Commons…")
    print("All works are pre-1900. Public domain worldwide.")
    print(f"Using Wikimedia Commons API to resolve thumbnail URLs ({THUMB_WIDTH}px)")
    download_all(only_style=args.style)
