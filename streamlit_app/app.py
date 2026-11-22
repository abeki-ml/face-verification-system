import io
import os
import uuid

import numpy as np
import psycopg
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client

import face_tools

load_dotenv()

THRESHOLD = 0.40


def secret(name):
    return os.getenv(name) or st.secrets[name]


DATABASE_URL = secret("DATABASE_URL")
supabase = create_client(secret("SUPABASE_URL"), secret("SUPABASE_SECRET_KEY"))


def upload_embedding(person_id, fingerprint):
    file_name = person_id + ".npy"

    buffer = io.BytesIO()
    np.save(buffer, fingerprint)
    data = buffer.getvalue()

    bucket = supabase.storage.from_("face-embeddings")
    bucket.upload(file_name, data, {"content-type": "application/octet-stream"})


def download_embedding(person_id):
    file_name = person_id + ".npy"

    bucket = supabase.storage.from_("face-embeddings")
    data = bucket.download(file_name)

    return np.load(io.BytesIO(data))


def save_user(person_id, name):
    with psycopg.connect(DATABASE_URL) as connection:
        connection.execute("INSERT INTO users (id, name) VALUES (%s, %s)", (person_id, name))


def get_users():
    with psycopg.connect(DATABASE_URL) as connection:
        result = connection.execute("SELECT id, name FROM users")
        return result.fetchall()

def fingerprint_of(photo):
    image = face_tools.read_image(photo.getvalue())
    if image is None:
        return None
    return face_tools.get_fingerprint(image)


def find_best_match(fingerprint):
    best = None
    for person_id, name in get_users():
        saved = download_embedding(person_id)
        score = float(np.dot(saved, fingerprint))
        if best is None or score > best["score"]:
            best = {"id": person_id, "name": name, "score": score}
    return best


def register(name, fingerprint):
    best = find_best_match(fingerprint)
    if best is not None and best["score"] >= THRESHOLD:
        return False, "This face is already registered as " + best["name"]

    person_id = uuid.uuid4().hex[:8]
    upload_embedding(person_id, fingerprint)
    try:
        save_user(person_id, name)
    except Exception:
        supabase.storage.from_("face-embeddings").remove([person_id + ".npy"])
        return False, "Registration failed. Please try again."
    return True, name + " is registered"


def verify(fingerprint):
    best = find_best_match(fingerprint)
    if best is None:
        return False, "Nobody is registered yet"
    if best["score"] >= THRESHOLD:
        return True, "Welcome, " + best["name"] + "!  (score " + str(round(best["score"], 2)) + ")"
    return False, "Face not recognised  (score " + str(round(best["score"], 2)) + ")"



st.title("Face ID")

name = st.text_input("Your name (only needed to register)")
photo = st.camera_input("Look at the camera, then take a picture")

register_clicked = st.button("Register")
verify_clicked = st.button("Verify")

if register_clicked or verify_clicked:
    if photo is None:
        st.warning("Take a picture first.")
    elif register_clicked and name.strip() == "":
        st.warning("Type your name first.")
    else:
        with st.spinner("Working..."):
            fingerprint = fingerprint_of(photo)
            if fingerprint is None:
                ok, message = False, "No face found. Look at the camera and try again."
            elif register_clicked:
                ok, message = register(name.strip(), fingerprint)
            else:
                ok, message = verify(fingerprint)

        if ok:
            st.success(message)
        else:
            st.error(message)