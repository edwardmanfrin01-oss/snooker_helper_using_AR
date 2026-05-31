# snooker_helper_using_AR
Implementation of a system for evaluating the trajectory of a shot, with the purpose of helping inexperienced players.


\------ Snooker helper using augmented reality ------

How it works:

1. in the code "config.py", you can set the OUTPUT\_DIR where all the clips, results and files will be stored, based on a datetime name for the folder

2. Generate Aruco markers using the code "aruco\_generator.py", and print them, they must be put on the corners of the pool and on the cue

3. Calibrate the scene, using the code "calibrate\_scene.py", obtaining the homography in the file "homography.npy"

4. Using the code "clips\_generator.py", do exactly one snapshot of the empty table(to be used for BG removal), and some recordings for testing

5. In the code "v13-trajectory-colordiff.py", set CLIP value to select which clip to use (or set "None" to use live webcam. Once the capture starts, you can:

   * Press SPACEBAR to obtain the AR overlay
   * Press 'r' to record a clip
   * Press 's' to save a snapshot for each of the windows



The latest version is "v13-trajectory-colordiff.py". This is the only one fully implemented, older versions are still available for possible future developments. They are all working.

