@echo off
echo Setting up vtuber-asr-automation...
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
echo
pause