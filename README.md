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
https://user-images.githubusercontent.com/75153467/176688080-03ca71a8-c99b-473e-ae48-b018930bd3e1.mp4
#### Photo
![Salman](https://user-images.githubusercontent.com/75153467/176688380-7c1495dd-13de-4048-a93b-5e05f974575c.jpeg)
#### Face Swapped Result
https://user-images.githubusercontent.com/75153467/176688421-54192f83-bf54-48fb-bcd9-0594752e3f4b.mp4
