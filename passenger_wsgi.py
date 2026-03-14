"""
SiteGround / Phusion Passenger entry point.
This file tells SiteGround how to start the Flask app.
"""
import os
import sys

# Make sure Python can find app.py in this same directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set your environment variables here if you prefer not to use
# SiteGround's UI (replace the placeholder values):
# os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-...")
# os.environ.setdefault("ADMIN_PASSWORD",    "yourpassword")
# os.environ.setdefault("BOT_NAME",          "Debate Assistant")

from app import app as application  # "application" is required by Passenger
