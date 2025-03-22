from flask import Flask, render_template, request, redirect, flash
import psycopg2
from dotenv import load_dotenv
import os
import re

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")  # for flashing messages

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

EMAIL_REGEX = re.compile(r"[^@]+@[^@]+\.[^@]+")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/waitlist", methods=["POST"])
def waitlist():
    email = request.form.get("email", "").strip()
    if not email or not EMAIL_REGEX.match(email):
        flash("Invalid email address", "error")
        return redirect("/")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Assuming a table "waitlist" with column "email" exists.
        cur.execute("INSERT INTO waitlist (email) VALUES (%s)", (email,))
        conn.commit()
        cur.close()
        conn.close()
        flash("Thanks for joining the waitlist.", "success")
    except Exception as e:
        flash(f"Database error: {e}", "error")
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
