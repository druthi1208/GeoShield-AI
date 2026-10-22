# GeoShield AI — Road Damage Detection

GeoShield AI is an AI-powered road damage detection project designed to identify road surface defects from images using YOLO object detection and a Flask web application.

## Features

- Detects road damage in uploaded images.
- Uses a trained YOLO model to identify different damage classes.
- Displays detection results through a web interface.
- Includes a prototype risk classification feature.

## Technologies Used

- Python
- Flask
- Ultralytics YOLO
- OpenCV
- HTML, CSS, and JavaScript

## Project Structure

```text
GeoShield-AI/
├── notebook/
│   └── explore.ipynb
├── scripts/
│   └── voc_to_yolo.py
├── src/
│   └── main.py
├── web/
│   ├── app.py
│   └── templates/
│       └── index.html
└── .gitignore
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/druthi1208/GeoShield-AI.git
cd GeoShield-AI
```

### 2. Create and activate a virtual environment

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the Python packages required by the application. A dependency file will be added to the repository so the environment can be reproduced consistently.

### 4. Configure the model and dataset

The trained model weights and dataset are not included in this repository. Obtain the required files separately and place them in the paths expected by the application before running it.

### 5. Run the application

Follow the setup instructions for the Flask application in `web/app.py`. Once configured, open the local address printed by Flask in your browser.

## Risk Classification

The project includes a prototype risk classification based on detected damage. This is an experimental feature and should not be treated as a validated road safety assessment.

## Future Improvements

- Improve detection accuracy across road conditions.
- Add evaluation metrics and example results.
- Improve risk classification using validated criteria.
- Add reproducible installation and deployment instructions.

## Author

**Druthi**

## License

A license has not yet been specified for this project.
