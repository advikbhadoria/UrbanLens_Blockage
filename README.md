# UrbanLens: Road Blockage Detection Prototype

An edge-based computer vision system designed to run on transit vehicles (such as city buses) for real-time road hazard and blockage detection. The system detects physical obstructions—fallen trees, utility poles, temporary construction zones, and misplaced barricades—and routes coordinates to civic authorities and navigation services for proactive traffic rerouting.

============================================================
PROJECT SETUP & ENVIRONMENT CONFIGURATION
============================================================

Follow these step-by-step setup instructions to configure your local machine:

1. Clone the repository:
   git clone https://github.com/advikbhadoria/UrbanLens_Blockage.git
   cd UrbanLens_Blockage

2. Set up an isolated Python virtual environment:
   - On Windows (Command Prompt):
     python -m venv venv
     venv\Scripts\activate

   - On macOS / Linux:
     python3 -m venv venv
     source venv/bin/activate

3. Install required dependencies:
   pip install --upgrade pip
   pip install ultralytics opencv-python fastapi uvicorn torch torchvision

============================================================
HOW TO RUN THE PROTOTYPE (ALL COMMANDS)
============================================================

Run each command in order from your terminal or Command Prompt inside the project directory:

Command 1: Navigate into the repository folder
cd UrbanLens_Blockage

Command 2: Activate the virtual environment
- If you are on Windows:
venv\Scripts\activate

- If you are on macOS / Linux:
source venv/bin/activate

Command 3: Confirm the required model and script files exist
- On Windows:
dir best.pt run_detector.py

- On macOS / Linux:
ls best.pt run_detector.py

Command 4: Run the blockage detection script
python run_detector.py

Command 5: Close and exit the detection window
Click on the OpenCV video display window and press:
q

Command 6: Deactivate the virtual environment when finished
deactivate