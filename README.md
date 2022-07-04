# Face Swap
A face swapping tool, to swap the face in a video with the face in a photo. A folder of photos, where each photo swaps over the folder of videos, outputs the face swapped videos.
If you are a developer, you are encouraged to go through the code.

[Google Colab](https://colab.research.google.com/drive/1YA_bHKExdEzaCjAAPVyeVfM09pX5PntP?usp=sharing)

## Packages Installation
- [cv2](https://pypi.org/project/opencv-python/)
- numpy (by default, gets installed with opencv)
- [dlib](https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-10-57348ba1117f)
- [moviepy](https://pypi.org/project/moviepy/)
- os (already comes preinstalled)
- glob (already comes preinstalled)

## How to Use
1. Clone the repository in your pc.
2. The 'photos' and 'videos' folder, and "shape_predictor_68_face_landmarks.dat" file should be present in the same directory as that of the "face swap.py" file.
3. Put some photos in the 'photos' folder and some videos in the 'videos' folder. **Remember that the video must be in the .mp4 format**. It is suggested to use .jpg format as the file container of the photos.
4. Open the folder in your python ide and run the code. Wait for sometime and have a glass of water😀.

## Result
After the code is executed, the result folder will be created in the same directory, containing a bunch of face swapped videos, made by the swapping of the face in each video, with the face in each photo. If we have used 4 photos and 3 videos, the "result" folder will contain 4*3, i.e., 12 face swapped videos.

## Some Results
#### Video
https://user-images.githubusercontent.com/75153467/177191306-5b5efeee-23d9-438d-9381-7ce733eba403.mp4
#### Photo
![photo](https://user-images.githubusercontent.com/75153467/177191331-291eb75d-9a6b-41f9-9dd8-4734d0751c7e.jpeg)
#### Face Swapped Result
https://user-images.githubusercontent.com/75153467/177191367-2dd4b658-9384-462f-91a1-97534a3b8922.mp4


