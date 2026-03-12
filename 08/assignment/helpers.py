"""Display helpers for the Murder Mystery Agents assignment.

Rich HTML display functions that make investigation output visually engaging
in Jupyter notebooks — styled evidence cards, chat bubbles, verdict panels,
and detective notes.
"""

from IPython.display import display, HTML


# Emoji avatars for suspects — keyed by character ID
CABIN_EMOJIS = {
    "diana": "&#128131;",      # woman dancing (socialite)
    "larry": "&#129406;",      # hiking boot (outdoorsman)
    "tom": "&#129658;",        # stethoscope (pediatrician)
    "sofia": "&#128247;",      # camera (photographer)
    "marcus": "&#128128;",     # skull (victim)
}
HOSPITAL_EMOJIS = {
    "dr_blake": "&#129656;",   # heart (cardiologist)
    "nurse_chen": "&#128137;", # syringe (nurse)
    "dr_santos": "&#127973;",  # hospital (ER doc)
    "orderly_james": "&#128295;",  # wrench (orderly)
    "dr_voss": "&#128128;",   # skull (victim)
}
LOCATION_EMOJIS = {
    "Living Room": "&#128715;",     # couch
    "Kitchen": "&#127860;",         # fork and knife
    "Mudroom / Back Entry": "&#128694;", # door
    "Upstairs Hallway": "&#128682;",     # stairs
    "Back Porch": "&#127795;",      # tree
    "Marcus's Bedroom": "&#128719;",  # bed
}


def _emoji_for(name, *dicts):
    """Look up emoji for a character/location name across provided dicts."""
    key = name.lower().strip().replace(" ", "_").replace("'s", "").replace("'s", "")
    for d in dicts:
        if key in d:
            return d[key]
        for k, v in d.items():
            if k in key or key in k:
                return v
    return ""


def case_briefing(title, setting, victim_name, cause, suspects, locations=None):
    """Display a styled case briefing panel."""
    suspect_items = "".join(f"<li>{s}</li>" for s in suspects)
    loc_section = ""
    if locations:
        loc_items = "".join(f"<li>{l}</li>" for l in locations)
        loc_section = f"<div style='margin-top:8px'><b>Locations:</b><ul style='margin:4px 0'>{loc_items}</ul></div>"
    display(HTML(f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);color:#e0e0e0;padding:20px 24px;border-radius:10px;border-left:5px solid #c62828;font-family:Georgia,serif;margin:12px 0;box-shadow:0 2px 8px rgba(0,0,0,0.3)">
        <div style="font-size:20px;font-weight:bold;color:#ff8a80;letter-spacing:0.5px">&#128373; {title}</div>
        <div style="margin:10px 0;font-style:italic;color:#bbb;border-bottom:1px solid #333;padding-bottom:10px">{setting[:250]}</div>
        <div style="margin:8px 0"><b>Victim:</b> {victim_name} &mdash; <span style="color:#ef9a9a">{cause}</span></div>
        <div><b>Suspects:</b><ul style="margin:4px 0">{suspect_items}</ul></div>
        {loc_section}
    </div>"""))


def display_stage(number, title, description=""):
    """Display a stage header banner to visually separate investigation phases."""
    desc_html = f'<div style="color:#b0bec5;font-size:13px;margin-top:4px">{description}</div>' if description else ""
    display(HTML(f"""
    <div style="background:linear-gradient(90deg,#0d47a1,#1565c0,#1976d2);color:white;padding:12px 20px;border-radius:8px;margin:16px 0 8px 0;font-family:sans-serif;box-shadow:0 2px 6px rgba(0,0,0,0.2)">
        <span style="background:rgba(255,255,255,0.2);padding:2px 10px;border-radius:12px;font-size:12px;font-weight:bold;margin-right:10px">STAGE {number}</span>
        <span style="font-size:16px;font-weight:bold">{title}</span>
        {desc_html}
    </div>"""))


def display_search(location_name, clues):
    """Display a location search result as a styled evidence card."""
    emoji = _emoji_for(location_name, LOCATION_EMOJIS) or "&#128269;"
    clue_items = "".join(f"<li style='margin:2px 0'>{c}</li>" for c in clues)
    display(HTML(f"""
    <div style="background:#263238;color:#e0e0e0;padding:12px 16px;border-radius:8px;border-left:4px solid #ffd54f;margin:6px 0;font-family:'Courier New',monospace;font-size:13px">
        <div style="font-size:14px;margin-bottom:6px">{emoji} <b style="color:#ffd54f">{location_name}</b></div>
        <ul style="margin:0;padding-left:20px;color:#cfd8dc">{clue_items}</ul>
    </div>"""))


def display_interrogation(name, question, response, emoji=""):
    """Display an interrogation exchange as styled chat bubbles."""
    if not emoji:
        emoji = _emoji_for(name, CABIN_EMOJIS, HOSPITAL_EMOJIS)
    suspect_label = f"{emoji} <b>{name}</b>" if emoji else f"<b>{name}</b>"
    display(HTML(f"""
    <div style="margin:8px 0;font-family:sans-serif;font-size:13px">
        <div style="display:flex;align-items:flex-start;margin-bottom:4px">
            <div style="background:#1b5e20;color:#e8f5e9;padding:8px 14px;border-radius:14px 14px 14px 2px;max-width:85%">
                <div style="font-size:11px;color:#a5d6a7;margin-bottom:2px">&#128373; <b>Detective</b> &rarr; <i>{name}</i></div>
                {question}
            </div>
        </div>
        <div style="display:flex;justify-content:flex-end">
            <div style="background:#37474f;color:#eceff1;padding:8px 14px;border-radius:14px 14px 2px 14px;max-width:85%">
                <div style="font-size:11px;color:#90a4ae;margin-bottom:2px">{suspect_label}</div>
                {response}
            </div>
        </div>
    </div>"""))


def display_notes(title, items):
    """Display investigator notes/takeaways between stages as a styled panel."""
    if isinstance(items, dict):
        body = "".join(
            f"<div style='margin:4px 0'><span style='color:#81d4fa;font-weight:bold'>{k}:</span> "
            f"{'<ul style=\"margin:2px 0 2px 16px;padding:0\">' + ''.join(f'<li>{i}</li>' for i in v) + '</ul>' if isinstance(v, list) else f' {v}'}</div>"
            for k, v in items.items()
        )
    else:
        body = "".join(f"<div style='margin:2px 0'>- {item}</div>" for item in items)
    display(HTML(f"""
    <div style="background:#1a237e;color:#e0e0e0;padding:14px 18px;border-radius:8px;border-left:4px solid #42a5f5;margin:8px 0;font-family:sans-serif;font-size:13px">
        <div style="font-size:14px;font-weight:bold;color:#90caf9;margin-bottom:6px">&#128221; {title}</div>
        {body}
    </div>"""))


def display_witness(name, role, statement, emoji=""):
    """Display a witness statement from the hospital mystery."""
    if not emoji:
        emoji = _emoji_for(name, HOSPITAL_EMOJIS, CABIN_EMOJIS)
    name_label = f"{emoji} <b>{name}</b>" if emoji else f"<b>{name}</b>"
    display(HTML(f"""
    <div style="background:#37474f;color:#eceff1;padding:10px 14px;border-radius:8px;border-left:4px solid #80cbc4;margin:6px 0;font-family:sans-serif;font-size:13px">
        <div style="font-size:11px;color:#80cbc4;margin-bottom:4px">&#128483; WITNESS STATEMENT</div>
        {name_label} <span style="color:#90a4ae">({role})</span>
        <div style="margin-top:4px;font-style:italic;color:#cfd8dc">"{statement}"</div>
    </div>"""))


def display_verdict(label, fields):
    """Display a final accusation/solution as a dramatic verdict panel."""
    rows = "".join(
        f"<tr><td style='padding:6px 14px;color:#ffab91;font-weight:bold;vertical-align:top;white-space:nowrap'>{k}</td>"
        f"<td style='padding:6px 14px;color:#e0e0e0'>{v}</td></tr>"
        for k, v in fields.items()
    )
    display(HTML(f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#0d1117);color:#e0e0e0;padding:20px 24px;border-radius:10px;border-left:5px solid #c62828;font-family:Georgia,serif;margin:16px 0;box-shadow:0 2px 8px rgba(0,0,0,0.3)">
        <div style="font-size:18px;font-weight:bold;color:#ff8a80;margin-bottom:10px;letter-spacing:1px">&#9878; {label}</div>
        <table style="border-collapse:collapse">{rows}</table>
    </div>"""))
