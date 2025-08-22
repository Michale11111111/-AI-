# ZBM Project

This project contains files related to human detection and interception, likely for a system involving human interaction or control.

## Files Overview

- `install-interception.exe`: An executable file possibly used for installing or setting up the interception system.
- `multithreaded_human_detection.py`: A Python script for human detection, likely utilizing multithreading for improved performance. This suggests it processes video streams or images to identify human presence.
- `yolov8n-pose.pt`: A pre-trained model file, specifically a YOLOv8 nano model for pose estimation. This is used by the human detection script to identify and track human poses.
- `x64/`:
  - `interception.dll`: A Dynamic Link Library (DLL) file, likely a core component of the interception system, providing functionalities for intercepting inputs or events.
  - `interception.lib`: A library file associated with `interception.dll`, used during the compilation of applications that link against the interception system.

## Usage

To use this project, you would typically:

1. **Install the interception system**: Run `install-interception.exe` to set up the necessary drivers or services.
2. **Run the human detection script**: Execute `multithreaded_human_detection.py` to start the human detection process. Ensure that `yolov8n-pose.pt` is in the same directory or accessible path.

Further details on specific configurations or dependencies might be required depending on the exact implementation of the scripts and libraries.