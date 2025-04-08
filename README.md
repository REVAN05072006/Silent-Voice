# Silent-Voice
This project is a real-time sign language to text translator that bridges communication gaps for the deaf and hard-of-hearing community. Using computer vision and machine learning, it detects hand gestures from a live camera feed and converts them into readable text, enabling seamless interaction without interpreters.

# Project Overview
The Sign Language to Text Converter is an innovative application designed to bridge communication barriers for the deaf and hard-of-hearing community. By leveraging computer vision and machine learning, this system interprets hand gestures from a live video feed and translates them into readable text or spoken words in real time. The goal is to create an accessible, intuitive tool that fosters seamless interaction between sign language users and those who may not understand sign language.

# Technical Approach
At its core, the project utilizes MediaPipe for accurate hand tracking and landmark detection, combined with a deep learning model (such as a CNN or LSTM) trained to classify ASL (American Sign Language) gestures. The pipeline begins with OpenCV capturing video frames, which are then processed to isolate hand movements. These gestures are fed into the trained model, which predicts the corresponding letters or words. For enhanced usability, the system can integrate text-to-speech (TTS) functionality, enabling audible output alongside visual text.

# Implementation Details
The application is built using Python, with OpenCV handling real-time video processing and MediaPipe extracting key hand landmarks. The classification model is trained on a dataset of ASL gestures—either custom-recorded or sourced from publicly available datasets like the ASL Alphabet dataset on Kaggle. The frontend can be implemented as a lightweight GUI (using Tkinter or PyQt) or a web interface (via Flask), depending on deployment needs. Performance optimizations, such as frame skipping or model quantization, ensure smooth real-time operation even on modest hardware.

# Impact and Future Work
This project has significant potential in education, healthcare, and everyday communication, offering an affordable alternative to expensive proprietary systems. Future enhancements could include expanding the gesture vocabulary to cover more phrases, supporting regional sign languages, or integrating with mobile devices for portability. Contributions from the open-source community are encouraged to refine accuracy, usability, and scalability.
