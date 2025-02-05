# EEG_based_drone_navigation_implementation

Author: Nachiketh G

This project is not a real-time implementation; rather, the drone operates based on movement signals predicted by a Brain-Computer Interface (BCI) model. The model is trained on EEG signals corresponding to drone movements using Motor Imagery.

## Modeling Approach:

- SVM Model: Used to evaluate the accuracy of 18 EEG electrodes from the MITASR headset and identify the best 4 electrodes for classification.
- Random Forest Model: Predicts the event column, which represents different drone movements.
- The predicted drone movement commands are then sent to an ESP32 to control the drone.

