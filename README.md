# Classroom Atmosphere Assessment Dataset（CAA）

About the dataset
Our CAA includes three modes: video, audio, and text, covering 9 subjects.
The original dataset consists of 246 videos, of which 198 are 20 minute videos and 48 are 13-15 minute videos. All videos have a resolution of 1920 × 1080, with a frame rate of 30fps.
Each video contains 4 feature labels and 9 feature data of three modalities. The four feature labels are subject, classroom atmosphere score, classroom atmosphere binarization (pass: 1; failed: 0), and regression score level (there are five levels here, namely Excellent: 90-100; Good: 80-89; Medium: 70-79; Poor: 60-69; Failed: 0-59). Nine feature data: 
                    visual:XXX_feature.txt、XXX_feature3D.txt、XXX_gaze.txt、XXX_pose.txt、XXX_aus.txt、XXX_hog.bin
                    Audio: XXX_covarep.csv、XXX_ formant.csv
                    Text: XXX_ Transcript.csv

Our dataset contains 246 folders from 101 to 346. Each video is grouped by a folder named Video Number. Among them, 197 were used as training sets and 49 were used as test sets.


train_ split. csv: This file contains the video ID, teacher gender, binary label (pass=1; failed=0), classroom atmosphere score, and teaching subjects (including Chinese, Mathematics, English, Physics, Chemistry, History, Geography, Biology, and Politics).
test_ split. csv: This file contains the video ID, teacher gender,binary label (pass=1; failed=0), classroom atmosphere score, and teaching subjects.

Each session folder contains the following files, using video number 101 as an example:


1. Visual files
T. Baltrušaitis, P. Robinson, L-P. Morency. OpenFace：An open source facial behavior analysis toolkit.2016 IEEE Winter Computer Vision Applications Conference (WACV).
http://ieeexplore.ieee.org/abstract/document/7477553/
Link: https://github.com/TadasBaltrusaitis/OpenFace

file：
（1）101_aus.txt :
The facial action unit has the following file format:
“frame, timestamp, confidence, success, AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU09_r, AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU25_r, AU26_r, AU04_c, AU12_c, AU15_c, AU23_c, AU28_c, AU45_c”.The value represented by ''r' is the regression output of each action unit, while '_c' reflects the presence (1) or absence (0) of a binary label for an action unit.
paper:RUNNING HEAD:Facial Action Coding System
https://en.wikipedia.org/wiki/Facial_Action_Coding_System

（2）101_feature.txt:
The file format of the 68 2D key points on the face is as follows:
“frame, timestamp(seconds), confidence, detection_success, x0, x1,…, x67, y0, y1,…,y67”.points are represented in pixel coordinates.

（3）101_feature3D.txt:
The file format of the 68 3D key points on the face is as follows:
“frame, timestamp(seconds), confidence, detection_success, X0, X1,…, X67, Y0, Y1,…, Y67, Z0, Z1,…, Z67”.These points are measured in millimeters in world coordinate space, with the camera located at (0,0,0) and the axis aligned with the camera.


（4）101_gaze.txt:
The file format for staring at the target location is as follows:
“frame, timestamp(seconds), confidence, detection_success, x_0, y_0, z_0, x_1, y_1, z_1, x_h0, y_h0, z_h0, x_h1, y_h1, z_h1”.The focus output is 4 vectors, with the first two vectors describing the focus direction of the two eyes in the world coordinate space, and the last two vectors describing the head coordinate space (therefore, if the eyes roll up, that is, the head rotates or tilts).

（5）101_pose.txt:
The file format for eye movement positions is as follows:
“frame_number, timestamp(seconds), confidence, detection_success, X, Y, Z, Rx, Ry, Rz”.The output includes 6 numbers, where X, Y, and Z are position coordinates, and Rx, Ry, and Rz are head rotation coordinates. The position is in millimeters in world coordinates, and the rotation is in radians and Euler angles (to get an appropriate rotation matrix, use R=Rx * Ry * Rz).

（6）101_hog.bin:
Using Felzenswalb's HoG to display the HOG face in binary file format on the aligned 112×112 area results in 4464 vectors per frame. Its storage method is that each frame of the byte stream is: "num_cols, num_rows, num_channels, valid_frame, 4464d vector". The "Read_HOG_files. m" framework from CLM reads the HOG binary format into the Matlab matrix.

Note that all. txt files contain appropriate titles. Each row represents the result of one frame.
"Confidence" is a measure in [0,1] that represents the confidence level of tracking.

2. Audio files
All audio formats are mono, with a frequency of 16kHz. Audio files may contain a small amount of noise; Use text transcription files to alleviate.

The audio function is extracted using the COVAREP toolbox located at: https://github.com/covarep/covarep
file:
(1)101_ covarep.csv (scrubbed): Extracts the following functions:
①All features are sampled at 10 millisecond intervals.
②F0, VUV, NAQ, QOQ, H1H2, PSP, MDQ, peakSlope, Rd, Rd_conf, MCEP_0-24、HMPDM_0-24, HMPDD_0-12.
③ You can find descriptions of each feature on the COVAREP website, as well as in the provided COVAREP publications. In addition, for detailed information on the exact steps of feature extraction, please refer to the publications referenced in the COVAREP script provided through github.
④ Note that if the VUV (sound/sound) provides a marker ({0,1}), the current segment is either turbid or clear. In the case of voiceless, i.e. VUV=0, the vocal folds do not vibrate.
(2)101_formant.csv (scrubbed)
① Including the first 5 resonance peaks, namely the vocal tract.
② Resonance frequency.

3. Text Transcription File
101_Transcript.csv (scrubbed)
The file includes timestamp and teacher's voice transcript content, removing voice content other than the teacher and cleaning the transcript file to remove some noise.

If you use the dataset, please cite this paper:
CAA: A Multimodal Dataset for Classroom Atmosphere Assessment in Real-world Situations.
