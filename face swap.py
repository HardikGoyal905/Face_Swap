import os
from moviepy.editor import *
import cv2
import dlib
import numpy as np
from glob import glob

photos = glob("photos/*") # returns a list of locations of the photos in the photos folder
videos = glob("videos/*")
result = "result"
landmark_prediction_model="shape_predictor_68_face_landmarks.dat"

trash = "trash" # This is acting as the location of a trash folder which we will be using and then deleting in this project NOTE: CHANGE THE NAME OF IT, IF YOU HAVE AN ANOTHER FOLDER NAMED TRASH IN THE DIRECTORY

detector = dlib.get_frontal_face_detector() # detector is the function which will detect the face
predictor = dlib.shape_predictor(landmark_prediction_model) # this function will predict the landmark points on the face, and the argument of shape_predictor is the 

# making the folder trash and temporarily using as our output folder for now
if not os.path.exists(trash):
    os.makedirs(trash)

if not os.path.exists(result):
    os.makedirs(result)


def main(img, img2, landmark_pts, triangle_index):
    ### Doing the same process for the frame image as done with the photo ###
    faces2 = detector(img2)

    landmark_pts2 = []

    for face in faces2:
        landmark2 = predictor(img2, face)

        for n in range(0, 68):
            x = landmark2.part(n).x
            y = landmark2.part(n).y

            landmark_pts2.append((x, y))
    
        break

    pts2 = np.array(landmark_pts2)

    hull2 = cv2.convexHull(pts2)

    # finding center of the face in the frame
    rectt = cv2.boundingRect(pts2)
    (xx, yy, ww, hh) = rectt
    center2 =(int((xx + xx + ww) / 2), int((yy + yy +hh) / 2))

    new_face = np.zeros_like(img2)

    ### finding the triangle in the frame corresponding to the triangle in the photo in terms of index of landmark points ###

    for t in triangle_index:
        # for each triangle in the photo

        # taking out each coordinate of triangle
        tr1_pt1 = landmark_pts[t[0]]
        tr1_pt2 = landmark_pts[t[1]]
        tr1_pt3 = landmark_pts[t[2]]

        t1_pts = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

        rect = cv2.boundingRect(t1_pts) # rectangle enclosing that triangle
        (x, y, w, h) = rect

        cropped = img[y: y+h, x: x+w] # cropping that rectangle

        mask = np.zeros_like(cropped)

        cropped_pts = np.array([[tr1_pt1[0]-x, tr1_pt1[1]-y],
                                [tr1_pt2[0]-x, tr1_pt2[1]-y],
                                [tr1_pt3[0]-x, tr1_pt3[1]-y]], np.int32)

        cv2.fillConvexPoly(mask, cropped_pts, (255, 255, 255)) # mask for extracting that triangle


        # For the corresponding triangle in the frame
        tr2_pt1 = landmark_pts2[t[0]]
        tr2_pt2 = landmark_pts2[t[1]]
        tr2_pt3 = landmark_pts2[t[2]]

        t2_pts = np.array([[tr2_pt1, tr2_pt2, tr2_pt3]], np.int32)

        rect2 = cv2.boundingRect(t2_pts)

        (x2, y2, w2, h2) = rect2

        cropped2 = img2[y2: y2+h2, x2: x2+w2]

        mask2 = np.zeros_like(cropped2)

        cropped2_pts = np.array([[tr2_pt1[0]-x2, tr2_pt1[1]-y2],
                                [tr2_pt2[0]-x2, tr2_pt2[1]-y2],
                                [tr2_pt3[0]-x2, tr2_pt3[1]-y2]], np.int32)

        cv2.fillConvexPoly(mask2, cropped2_pts, (255, 255, 255)) # mask for extracting that triangle

        ### Warped the triangle in the photo to the size of the corresponding triangle in the frame ###

        cropped_pts = np.float32(cropped_pts)
        cropped2_pts = np.float32(cropped2_pts)

        M = cv2.getAffineTransform(cropped_pts, cropped2_pts) # returns a transformation matrix (M) for warping

        t_warped = cv2.warpAffine(cropped, M, (w2, h2)) # warping the triangle

        t_warped = cv2.bitwise_and(t_warped, mask2) # Extracting the warped triangle

        ### Putting the warped triangular piece to make the new face ###

        triangle_area = new_face[y2: y2 + h2, x2: x2 + w2]

        """Due to slight deformities in the coordinates of the warped triangle, might arise between the operations, the triangles would be overlapping on each other.
        However this deformity is very small, but it would be visible to naked eyes when the final face would be created.
        So its better to cut out the overlapped region."""
        # Cutting out that overlap
        triangle_area_gray = cv2.cvtColor(triangle_area, cv2.COLOR_BGR2GRAY)
        _, triangle_mask = cv2.threshold(triangle_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        t_warped = cv2.bitwise_and(t_warped, t_warped, mask=triangle_mask)

        # adding the new triangle in the new face
        triangle_area = cv2.add(triangle_area, t_warped)
        new_face[y2: y2 + h2, x2: x2 + w2] = triangle_area

    #Making different masks
    new_face_gray = cv2.cvtColor(new_face, cv2.COLOR_BGR2GRAY)
    _, mask2 = cv2.threshold(new_face_gray, 1, 255, cv2.THRESH_BINARY_INV)
    ret, background = cv2.threshold(new_face_gray, 1, 255, cv2.THRESH_BINARY)

    # Making the final photo by inserting the new face on the old face of the frame
    no_face = cv2.bitwise_and(img2, img2, mask=mask2)
    resulting = cv2.add(new_face, no_face)

    # seamless cloning to give the final touch, so that the final image will look like original in context of skin tone, and other quality effects
    seamlessclone = cv2.seamlessClone(resulting, img2, background, center2, cv2.NORMAL_CLONE)

    return seamlessclone


if __name__ == "__main__":
    for photo in photos: # iterating through each photo

        ### reading the photo ###

        photo_name = os.path.basename(photo).split('.')[0] # name of the photo

        img = cv2.imread(photo) # reading the photo

        ### detecting the image in the photo and store the landmark points ###

        faces = detector(img) # detect the faces in the image

        landmark_pts = [] 

        triangle_index = []

        for face in faces:
            landmark = predictor(img, face)

            for n in range(0, 68):
                x = landmark.part(n).x
                y = landmark.part(n).y

                landmark_pts.append((x, y)) # storing the coordinates of landmark points (68 points in total)

            pts = np.array(landmark_pts) # converting to np array, which suits perfectly for open-cv library

            ### getting out delaunay triangles in the photo ###

            hull = cv2.convexHull(pts) # finding the convex hull of those points

            mask = np.zeros_like(img)

            cv2.fillConvexPoly(mask, hull, (255, 255, 255))

            final_img = cv2.bitwise_and(img, mask)

            rect = cv2.boundingRect(hull) # rectangle which is enclosing the face

            subdiv = cv2.Subdiv2D(rect) # this is the plane which will be divided into triangles

            subdiv.insert(landmark_pts) # divided into triangles, with vertices as landmark points

            triangles = subdiv.getTriangleList() # returns the list, each element representing the the coordinates of a triangle
            triangles = np.array(triangles, dtype= int)

            ### storing the triangles in the index form, where the index is the index of the landmark point ###
            for t in triangles: # t is the list of coordinates of the delaunay triangle
                ### working on each triangle ###
                # pti denotes the coordinates of ith point of triangle
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])

                # finding out which index of the landmark point is that coordinate of triangle points
                i1 = np.where((pts == pt1).all(axis=1))
                i2 = np.where((pts == pt2).all(axis=1))
                i3 = np.where((pts == pt3).all(axis=1))

                ind1 = None
                ind2 = None
                ind3 = None

                for n in i1[0]:
                    ind1 = n

                for n in i2[0]:
                    ind2 = n

                for n in i3[0]:
                    ind3 = n

                if ind1 is not None and ind2 is not None and ind3 is not None:
                    triangle_index.append((ind1, ind2, ind3)) #triangle_index is the list, with each element being a tuple, whose each element represents which index of the landmark points does this triangular point resembles

            break # as we want only one face to swap on a video

        for video in videos:
            ### reading the video, the video should be mp4 ###
            video_name = os.path.basename(video).split('.')[0]

            cap = cv2.VideoCapture(video)
            # dimensions of the video
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            size = (int(w), int(h))

            path = os.path.join(trash, f'{video_name}_{photo_name}.avi') # path of resulting video
            # initialising the video writer which will write the final video with 'DIVX' as our video codec
            out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), cap.get(cv2.CAP_PROP_FPS), size)

            while True:
                ### working on each frame ###
                ret, frame = cap.read()

                if not ret:
                    cap.release()
                    break

                result_frame = main(img, frame, landmark_pts, triangle_index)

                out.write(result_frame)

            out.release()

            ### Adding original sound in our face swapped video ###
            clip = VideoFileClip(path)

            clip2 = VideoFileClip(video)

            clip.audio = clip2.audio

            ### Writing the final video in the 'result' folder
            clip.write_videofile(os.path.join(result, f'{video_name}_{photo_name}.mp4'))

            os.remove(path)

    if os.path.exists(trash):
        os.rmdir(trash)
