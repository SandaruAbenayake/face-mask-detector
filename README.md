😷 face-mask-detector
Real-time Face Mask Detection using MobileNetV2 (Transfer Learning)

This project uses a deep learning model trained on face mask images to detect whether a person is wearing a mask or not via webcam in real time.

✅ Features
 > Real-time mask detection using your laptop webcam

 > Trained using MobileNetV2 + transfer learning

 > Clean, step-by-step Jupyter workflow

 > Lightweight and fast model


🗂️ Project Structure

face-mask-detector-Repo/

├── jupyter_notebook/
     ── face_mask_detector.ipynb        # Jupyter Notebook with training steps

├── detect_mask_webcam.py               # Real-time mask detection script

├── best_model.h5                       # Trained model

├── kaggle_dataset_link                 #dataset

├── requirements.txt                    # Dependencies list

├── README.md                           # Project instructions



🔧 Setup Instructions (Command Prompt Style)


📌 Step 1: Create and Navigate to Project Folder

mkdir face-mask-detector

cd face-mask-detector


📌 Step 2: (Optional) Create Virtual Environment

python -m venv venv

venv\Scripts\activate


📌 Step 3: Clone or Copy Project Files

If using GitHub:

git clone https://github.com/SandaruAbenayake/face-mask-detector.git

cd face-mask-detector

If not, just manually copy your files (.ipynb, .h5, .py, etc.) into the folder.


📌 Step 4: Install Required Packages

pip install -r requirements.txt

If you don’t have a requirements.txt, install manually:

pip install tensorflow opencv-python matplotlib numpy


📌 Step 5: Launch Jupyter Notebook

jupyter notebook

Then open face_mask_detector.ipynb from the browser and run all cells.


📌 Step 6: Run Real-Time Webcam Detector

After training is done and best_model.h5 is saved:

python detect_mask_webcam.py

Press q to quit webcam window.






