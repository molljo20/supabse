import streamlit as st
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io
from supabase import create_client, Client
import datetime
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# Seitenkonfiguration mit Dark Mode
st.set_page_config(
    page_title="Fundbüro - KI-Erkennung",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark Mode CSS
st.markdown("""
<style>
    /* Dark Mode Styles */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Admin Panel Styling */
    .admin-panel {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #333;
        margin-bottom: 1rem;
    }
    
    .admin-header {
        color: #ff6b6b;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    /* Haupt-Buttons */
    .main-buttons {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .main-button {
        background-color: #2e2e2e;
        color: #fafafa;
        border: 2px solid #444;
        border-radius: 1rem;
        padding: 2rem 4rem;
        font-size: 2rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .main-button:hover {
        background-color: #3e3e3e;
        border-color: #666;
        transform: scale(1.05);
    }
    
    /* Fundstück Cards */
    .fund-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #2e2e2e;
        color: #fafafa;
        border: 1px solid #444;
    }
    
    .stButton button:hover {
        background-color: #3e3e3e;
        border-color: #666;
    }
    
    /* Delete Button */
    .delete-btn button {
        background-color: #ff4444;
        color: white;
    }
    
    .delete-btn button:hover {
        background-color: #ff6666;
    }
    
    /* Edit Button */
    .edit-btn button {
        background-color: #4444ff;
        color: white;
    }
    
    .edit-btn button:hover {
        background-color: #6666ff;
    }
    
    /* Input Fields */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background-color: #2e2e2e;
        color: #fafafa;
        border-color: #444;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e1e1e;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #fafafa;
    }
    
    /* Success/Error/Warning Messages */
    .stSuccess, .stError, .stWarning {
        background-color: #1e1e1e;
        border-color: #444;
    }
</style>
""", unsafe_allow_html=True)

# Supabase-Konfiguration
SUPABASE_URL = "https://imntylvenimvnmocbtzy.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImltbnR5bHZlbmltdm5tb2NidHp5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzMwNTk4NzcsImV4cCI6MjA4ODYzNTg3N30.48pIBqUdlqXTooorJXHm71icVSj1wdTwW4tg5m2ovns"

# Admin Passwörter
DELETE_PASSWORD = "6767"
EDIT_PASSWORD = "timgioh"

# Email-Konfiguration (ersetze mit deinen Daten)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "ohtimgi@gmail.com"
SMTP_PASSWORD = "ftqz vujw skbl bblu"

# Initialisiere Supabase Client
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# Lade das KI-Modell
@st.cache_resource
def load_keras_model():
    try:
        model = load_model("keras_model.h5", compile=False)
        class_names = open("labels.txt", "r").readlines()
        return model, class_names
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        return None, None

# Bild vorbereiten und klassifizieren
def prepare_and_classify(image, model, class_names):
    try:
        # Bild in RGB konvertieren
        image = image.convert("RGB")
        
        # Bild auf 224x224 zuschneiden
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        
        # Bild in Numpy-Array konvertieren
        image_array = np.asarray(image)
        
        # Normalisieren
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # Daten-Array erstellen
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # Vorhersage
        prediction = model.predict(data, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = float(prediction[0][index])
        
        # Bereinige den Klassennamen (entferne Nummerierung)
        if ': ' in class_name:
            class_name = class_name.split(': ')[1]
        
        return class_name, confidence_score, index
    except Exception as e:
        st.error(f"Fehler bei der Bildklassifizierung: {e}")
        return None, None, None

# Email senden
def send_email(to_email, found_item, description, location, image_url):
    try:
        # Email erstellen
        msg = MIMEMultipart()
        msg['From'] = SMTP_USERNAME
        msg['To'] = to_email
        msg['Subject'] = f"🔍 Mögliches Fundstück gefunden: {found_item}"
        
        # Email-Text
        body = f"""
        <h2>Ein mögliches Fundstück wurde gemeldet!</h2>
        
        <p><strong>Gegenstand:</strong> {found_item}</p>
        <p><strong>Beschreibung:</strong> {description}</p>
        <p><strong>Fundort:</strong> {location}</p>
        <p><strong>Bild:</strong> <a href="{image_url}">Hier klicken zum Ansehen</a></p>
        
        <p>Bitte überprüfe, ob dies dein verlorener Gegenstand sein könnte.</p>
        
        <p>Viele Grüße,<br>Dein KI-Fundbüro</p>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Email senden
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        return True
    except Exception as e:
        st.error(f"Fehler beim Senden der Email: {e}")
        return False

# Prüfe auf Übereinstimmungen mit gesuchten Gegenständen
def check_for_matches(supabase, new_item_class, new_item_description, new_item_image_url):
    try:
        # Hole alle gesuchten Gegenstände
        query = supabase.table("gesuchte_gegenstaende").select("*").execute()
        matches = []
        
        for searched in query.data:
            # Prüfe ob die Kategorie übereinstimmt
            if searched['class_name'].lower() in new_item_class.lower() or new_item_class.lower() in searched['class_name'].lower():
                # Prüfe ob Schlüsselwörter in der Beschreibung vorkommen
                keywords = searched['description'].lower().split()
                new_desc_lower = new_item_description.lower()
                
                match_score = 0
                for keyword in keywords:
                    if len(keyword) > 3 and keyword in new_desc_lower:
                        match_score += 1
                
                # Wenn Übereinstimmung gefunden (mindestens 1 Keyword oder gleiche Kategorie)
                if match_score > 0 or searched['class_name'].lower() == new_item_class.lower():
                    matches.append({
                        'email': searched['email'],
                        'item': searched['class_name'],
                        'description': searched['description'],
                        'match_score': match_score
                    })
        
        return matches
    except Exception as e:
        st.error(f"Fehler beim Prüfen auf Übereinstimmungen: {e}")
        return []

# Element in Supabase speichern
def save_to_supabase(supabase, image, class_name, confidence_score, description, location, finder_name):
    try:
        # Bild in Bytes konvertieren für Supabase
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Eindeutigen Dateinamen erstellen
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"fundstuecke/{timestamp}_{class_name}.png"
        
        # Bild in Supabase Storage hochladen
        supabase.storage.from_("fundbuero-bilder").upload(
            file_name, 
            img_byte_arr
        )
        
        # Öffentliche URL für das Bild generieren
        image_url = f"{SUPABASE_URL}/storage/v1/object/public/fundbuero-bilder/{file_name}"
        
        # Datensatz in der Datenbank erstellen
        data = {
            "class_name": class_name,
            "class_index": int(confidence_score * 100),
            "confidence_score": confidence_score,
            "description": description,
            "location": location,
            "finder_name": finder_name,
            "image_url": image_url,
            "created_at": datetime.datetime.now().isoformat(),
            "status": "gemeldet"
        }
        
        result = supabase.table("fundstuecke").insert(data).execute()
        
        # Prüfe auf Übereinstimmungen mit gesuchten Gegenständen
        matches = check_for_matches(supabase, class_name, description, image_url)
        
        # Sende Emails bei Übereinstimmungen
        for match in matches:
            send_email(
                match['email'],
                class_name,
                description,
                location,
                image_url
            )
        
        if matches:
            st.info(f"📧 {len(matches)} Benachrichtigung(en) wurden versendet!")
        
        return True, result
    except Exception as e:
        st.error(f"Fehler beim Speichern in Supabase: {e}")
        return False, None

# Gesuchten Gegenstand speichern
def save_searched_item(supabase, class_name, description, email):
    try:
        data = {
            "class_name": class_name,
            "description": description,
            "email": email,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        result = supabase.table("gesuchte_gegenstaende").insert(data).execute()
        return True, result
    except Exception as e:
        st.error(f"Fehler beim Speichern des gesuchten Gegenstands: {e}")
        return False, None

# Fundstück löschen
def delete_fundstueck(supabase, item_id):
    try:
        result = supabase.table("fundstuecke").delete().eq("id", item_id).execute()
        return True, result
    except Exception as e:
        st.error(f"Fehler beim Löschen: {e}")
        return False, None

# Fundstück bearbeiten
def update_fundstueck(supabase, item_id, updated_data):
    try:
        result = supabase.table("fundstuecke").update(updated_data).eq("id", item_id).execute()
        return True, result
    except Exception as e:
        st.error(f"Fehler beim Bearbeiten: {e}")
        return False, None

# Fundstücke aus Supabase abrufen
def get_fundstuecke(supabase, filter_class=None, search_term=None):
    try:
        query = supabase.table("fundstuecke").select("*").order("created_at", desc=True)
        
        if filter_class and filter_class != "Alle":
            query = query.eq("class_name", filter_class)
            
        if search_term:
            query = query.ilike("description", f"%{search_term}%")
            
        result = query.execute()
        return result.data
    except Exception as e:
        st.error(f"Fehler beim Abrufen der Fundstücke: {e}")
        return []

# Gesuchte Gegenstände abrufen
def get_searched_items(supabase):
    try:
        result = supabase.table("gesuchte_gegenstaende").select("*").order("created_at", desc=True).execute()
        return result.data
    except Exception as e:
        st.error(f"Fehler beim Abrufen der gesuchten Gegenstände: {e}")
        return []

# Admin Panel
def show_admin_panel(supabase):
    with st.expander("👨‍🏫 Admin-Panel (Lehrer)", expanded=False):
        st.markdown('<div class="admin-panel">', unsafe_allow_html=True)
        st.markdown('<div class="admin-header">🔐 Admin-Bereich</div>', unsafe_allow_html=True)
        
        admin_password = st.text_input("Admin-Passwort", type="password", key="admin_password")
        
        if admin_password == DELETE_PASSWORD:
            st.success("✅ Lösch-Modus aktiviert")
            st.session_state['admin_mode'] = 'delete'
        elif admin_password == EDIT_PASSWORD:
            st.success("✅ Bearbeiten-Modus aktiviert (inkl. Löschen)")
            st.session_state['admin_mode'] = 'edit'
        elif admin_password:
            st.error("❌ Falsches Passwort")
            st.session_state['admin_mode'] = None
        
        if 'admin_mode' in st.session_state and st.session_state['admin_mode']:
            st.info(f"Aktiver Modus: **{st.session_state['admin_mode']}**")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Haupt-App
def main():
    st.title("🔍 KI-Fundbüro")
    
    # Supabase und Modell initialisieren
    supabase = init_supabase()
    model, class_names = load_keras_model()
    
    # Admin Panel anzeigen
    show_admin_panel(supabase)
    
    if model is None or class_names is None:
        st.error("Das KI-Modell konnte nicht geladen werden. Bitte überprüfe die Dateien 'keras_Model.h5' und 'labels.txt'.")
        return
    
    # Extrahiere Klassennamen aus labels.txt
    available_classes = ["Alle"]
    for line in class_names:
        if ': ' in line:
            available_classes.append(line.split(': ')[1].strip())
        else:
            available_classes.append(line.strip())
    
    # Große Haupt-Buttons
    st.markdown('<div class="main-buttons">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 FUNDSTÜCK MELDEN", key="btn_melden", use_container_width=True):
            st.session_state['app_mode'] = 'melden'
    
    with col2:
        if st.button("🔎 NACH VERLORENEM SUCHEN", key="btn_suchen", use_container_width=True):
            st.session_state['app_mode'] = 'suchen'
    
    with col3:
        if st.button("📋 GESUCHTE GEGENSTÄNDE", key="btn_gesucht", use_container_width=True):
            st.session_state['app_mode'] = 'gesucht'
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Standard-Modus setzen
    if 'app_mode' not in st.session_state:
        st.session_state['app_mode'] = 'melden'
    
    # Verschiedene Ansichten basierend auf ausgewähltem Modus
    if st.session_state['app_mode'] == 'melden':
        show_report_tab(supabase, model, class_names)
    elif st.session_state['app_mode'] == 'suchen':
        show_search_tab(supabase, class_names, available_classes)
    elif st.session_state['app_mode'] == 'gesucht':
        show_wanted_tab(supabase, class_names)

# Fundstück melden Tab
def show_report_tab(supabase, model, class_names):
    st.header("📤 Neues Fundstück melden")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Wähle ein Bild des gefundenen Gegenstands aus",
            type=["jpg", "jpeg", "png", "bmp"]
        )
        
        if uploaded_file is not None:
            # Bild anzeigen
            image = Image.open(uploaded_file)
            st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
    
    with col2:
        if uploaded_file is not None and st.button("🔍 Gegenstand erkennen", type="primary", use_container_width=True):
            with st.spinner("KI analysiert das Bild..."):
                image = Image.open(uploaded_file)
                class_name, confidence_score, index = prepare_and_classify(image, model, class_names)
                
                if class_name:
                    st.session_state['detected_class'] = class_name
                    st.session_state['detected_confidence'] = confidence_score
                    st.session_state['detected_image'] = image
                    
                    st.success(f"✅ Erkannt: **{class_name}**")
                    st.info(f"Konfidenz: {confidence_score:.2%}")
        
        # Formular für Funddetails
        if 'detected_class' in st.session_state:
            with st.form("fund_form"):
                st.subheader("Details zum Fundstück")
                
                detected_class = st.text_input("Erkannte Kategorie", 
                                               value=st.session_state['detected_class'],
                                               disabled=True)
                
                description = st.text_area("Beschreibung", 
                                          placeholder="z.B. Farbe, Marke, besondere Merkmale...")
                
                location = st.text_input("Fundort", 
                                        placeholder="Wo wurde der Gegenstand gefunden?")
                
                finder_name = st.text_input("Name des Finders (optional)")
                
                submitted = st.form_submit_button("📦 Fundstück speichern", use_container_width=True)
                
                if submitted:
                    if description and location:
                        with st.spinner("Speichere in Datenbank..."):
                            success, result = save_to_supabase(
                                supabase,
                                st.session_state['detected_image'],
                                detected_class,
                                st.session_state['detected_confidence'],
                                description,
                                location,
                                finder_name or "Anonym"
                            )
                            
                            if success:
                                st.success("✅ Fundstück erfolgreich gespeichert!")
                                # Session State zurücksetzen
                                del st.session_state['detected_class']
                                del st.session_state['detected_confidence']
                                del st.session_state['detected_image']
                                st.rerun()
                    else:
                        st.warning("Bitte fülle alle Pflichtfelder aus (Beschreibung und Fundort).")

# Suche Tab
def show_search_tab(supabase, class_names, available_classes):
    st.header("🔎 Nach verlorenen Gegenständen suchen")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        search_term = st.text_input("🔍 Suchbegriff", placeholder="z.B. 'rote Flasche'...")
    
    with col2:
        filter_class = st.selectbox("Kategorie filtern", available_classes)
    
    with col3:
        st.write("")  # Platzhalter für Alignment
        search_button = st.button("Suchen", type="primary", use_container_width=True)
    
    # Fundstücke anzeigen
    fundstuecke = get_fundstuecke(supabase, filter_class if filter_class != "Alle" else None, search_term)
    
    if fundstuecke:
        st.success(f"📊 {len(fundstuecke)} Fundstück(e) gefunden")
        
        # Grid-Layout für Fundstücke
        cols = st.columns(3)
        for idx, fund in enumerate(fundstuecke):
            with cols[idx % 3]:
                with st.container(border=True):
                    # Bild anzeigen
                    if fund.get('image_url'):
                        st.image(fund['image_url'], use_column_width=True)
                    
                    st.markdown(f"### {fund['class_name']}")
                    st.markdown(f"**Beschreibung:** {fund['description']}")
                    st.markdown(f"**Fundort:** {fund['location']}")
                    st.markdown(f"**Gemeldet von:** {fund['finder_name']}")
                    st.markdown(f"**Datum:** {fund['created_at'][:10]}")
                    
                    # Konfidenz als Fortschrittsbalken anzeigen
                    confidence = fund.get('confidence_score', 0)
                    st.progress(confidence, text=f"KI-Konfidenz: {confidence:.1%}")
                    
                    # Status anzeigen
                    status = fund.get('status', 'gemeldet')
                    if status == 'gemeldet':
                        st.caption("🟡 Noch nicht abgeholt")
                    else:
                        st.caption("✅ Bereits abgeholt")
                    
                    # Admin-Funktionen (Löschen/Bearbeiten)
                    if 'admin_mode' in st.session_state:
                        col_del, col_edit = st.columns(2)
                        
                        # Löschen-Button (in beiden Modi verfügbar)
                        with col_del:
                            if st.session_state['admin_mode'] in ['delete', 'edit']:
                                if st.button(f"🗑️ Löschen", key=f"del_{fund['id']}", use_container_width=True):
                                    success, _ = delete_fundstueck(supabase, fund['id'])
                                    if success:
                                        st.success("✅ Gelöscht!")
                                        st.rerun()
                        
                        # Bearbeiten-Button (nur im edit mode)
                        with col_edit:
                            if st.session_state['admin_mode'] == 'edit':
                                if st.button(f"✏️ Bearbeiten", key=f"edit_{fund['id']}", use_container_width=True):
                                    st.session_state['editing_item'] = fund
                                    st.rerun()
                    
                    # Bearbeitungsformular anzeigen
                    if 'editing_item' in st.session_state and st.session_state['editing_item']['id'] == fund['id']:
                        with st.form(key=f"edit_form_{fund['id']}"):
                            st.markdown("### ✏️ Eintrag bearbeiten")
                            
                            new_description = st.text_area("Beschreibung", value=fund['description'])
                            new_location = st.text_input("Fundort", value=fund['location'])
                            new_finder = st.text_input("Finder", value=fund['finder_name'])
                            new_status = st.selectbox("Status", ["gemeldet", "abgeholt"], 
                                                     index=0 if fund['status'] == 'gemeldet' else 1)
                            
                            col_save, col_cancel = st.columns(2)
                            
                            with col_save:
                                if st.form_submit_button("💾 Speichern", use_container_width=True):
                                    updated_data = {
                                        "description": new_description,
                                        "location": new_location,
                                        "finder_name": new_finder,
                                        "status": new_status
                                    }
                                    success, _ = update_fundstueck(supabase, fund['id'], updated_data)
                                    if success:
                                        st.success("✅ Aktualisiert!")
                                        del st.session_state['editing_item']
                                        st.rerun()
                            
                            with col_cancel:
                                if st.form_submit_button("❌ Abbrechen", use_container_width=True):
                                    del st.session_state['editing_item']
                                    st.rerun()
    else:
        st.info("😕 Keine Fundstücke gefunden. Versuche andere Suchbegriffe oder lade ein neues Fundstück hoch!")

# Gesuchte Gegenstände Tab
def show_wanted_tab(supabase, class_names):
    st.header("📋 Gesuchte Gegenstände")
    
    st.markdown("""
    <div style="background-color: #1e1e1e; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
        <h4>🔔 So funktioniert's:</h4>
        <p>Trage hier ein, wonach du suchst. Wenn jemand einen passenden Gegenstand findet, 
        bekommst du automatisch eine Email mit dem Vorschlag, ihn dir anzuschauen!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Formular für neuen gesuchten Gegenstand (nur im Edit-Mode sichtbar)
    if 'admin_mode' in st.session_state and st.session_state['admin_mode'] == 'edit':
        with st.form("wanted_form"):
            st.subheader("🔍 Neuen Gegenstand suchen")
            
            # Extrahiere Klassennamen für Dropdown
            class_options = []
            for line in class_names:
                if ': ' in line:
                    class_options.append(line.split(': ')[1].strip())
                else:
                    class_options.append(line.strip())
            
            selected_class = st.selectbox("Kategorie", class_options)
            description = st.text_area("Beschreibung des gesuchten Gegenstands", 
                                      placeholder="z.B. 'Rote Trinkflasche mit Aufkleber'")
            email = st.text_input("Email-Adresse für Benachrichtigungen")
            
            submitted = st.form_submit_button("📌 Gegenstand suchen", use_container_width=True)
            
            if submitted:
                if description and email:
                    success, _ = save_searched_item(supabase, selected_class, description, email)
                    if success:
                        st.success("✅ Gesuchter Gegenstand wurde registriert!")
                        st.rerun()
                else:
                    st.warning("Bitte fülle alle Felder aus!")
    
    # Gesuchte Gegenstände anzeigen (auch für normale Benutzer sichtbar)
    st.subheader("Aktuelle Suchanfragen")
    searched_items = get_searched_items(supabase)
    
    if searched_items:
        for item in searched_items:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{item['class_name']}**")
                    st.markdown(f"*{item['description']}*")
                    st.caption(f"Gesucht seit: {item['created_at'][:10]}")
                
                with col2:
                    # Email nur im Edit-Mode sichtbar
                    if 'admin_mode' in st.session_state and st.session_state['admin_mode'] == 'edit':
                        st.markdown(f"📧 {item['email']}")
                        
                        # Löschen-Button für Admins
                        if st.button(f"🗑️", key=f"del_wanted_{item['id']}"):
                            try:
                                supabase.table("gesuchte_gegenstaende").delete().eq("id", item['id']).execute()
                                st.rerun()
                            except:
                                pass
                    else:
                        st.markdown("🔒 Email nur für Admins sichtbar")
    else:
        st.info("📭 Noch keine gesuchten Gegenstände eingetragen.")

if __name__ == "__main__":
    main()
