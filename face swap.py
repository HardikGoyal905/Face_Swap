import os
from moviepy.editor import *
import cv2
import dlib
import numpy as np
from glob import glob

# swapped_face = "ayush.jpg"
#
# src = "height2"

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def main(img, img2, landmark_pts, triangle_index):

    faces2 = detector(img2)

    landmark_pts2 = []

    for face in faces2:
        landmark2 = predictor(img2, face)

        for n in range(0, 68):
            x = landmark2.part(n).x
            y = landmark2.part(n).y

            landmark_pts2.append((x, y))

    pts2 = np.array(landmark_pts2)

    hull2 = cv2.convexHull(pts2)

    rectt = cv2.boundingRect(pts2)

    (xx, yy, ww, hh) = rectt

    center2 =(int((xx + xx + ww) / 2), int((yy + yy +hh) / 2))

    new_face = np.zeros_like(img2)

    for t in triangle_index:
        tr1_pt1 = landmark_pts[t[0]]
        tr1_pt2 = landmark_pts[t[1]]
        tr1_pt3 = landmark_pts[t[2]]

        t1_pts = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

        rect = cv2.boundingRect(t1_pts)
        (x, y, w, h) = rect

        # cv2.line(img, tr1_pt1, tr1_pt2, (0, 0, 255), 1)
        # cv2.line(img, tr1_pt2, tr1_pt3, (0, 0, 255), 1)
        # cv2.line(img, tr1_pt3, tr1_pt1, (0, 0, 255), 1)

        cropped = img[y: y+h, x: x+w]

        mask = np.zeros_like(cropped)

        cropped_pts = np.array([[tr1_pt1[0]-x, tr1_pt1[1]-y],
                                [tr1_pt2[0]-x, tr1_pt2[1]-y],
                                [tr1_pt3[0]-x, tr1_pt3[1]-y]], np.int32)

        cv2.fillConvexPoly(mask, cropped_pts, (255, 255, 255))


        #Image 2
        tr2_pt1 = landmark_pts2[t[0]]
        tr2_pt2 = landmark_pts2[t[1]]
        tr2_pt3 = landmark_pts2[t[2]]

        t2_pts = np.array([[tr2_pt1, tr2_pt2, tr2_pt3]], np.int32)

        rect2 = cv2.boundingRect(t2_pts)

        (x2, y2, w2, h2) = rect2

        # cv2.line(img2, tr2_pt1, tr2_pt2, (255, 0, 0), 1)
        # cv2.line(img2, tr2_pt2, tr2_pt3, (255, 0, 0), 1)
        # cv2.line(img2, tr2_pt3, tr2_pt1, (255, 0, 0), 1)

        cropped2 = img2[y2: y2+h2, x2: x2+w2]

        mask2 = np.zeros_like(cropped2)

        cropped2_pts = np.array([[tr2_pt1[0]-x2, tr2_pt1[1]-y2],
                                [tr2_pt2[0]-x2, tr2_pt2[1]-y2],
                                [tr2_pt3[0]-x2, tr2_pt3[1]-y2]], np.int32)

        cv2.fillConvexPoly(mask2, cropped2_pts, (255, 255, 255))

        # Warped Triangle

        cropped_pts = np.float32(cropped_pts)
        cropped2_pts = np.float32(cropped2_pts)

        M = cv2.getAffineTransform(cropped_pts, cropped2_pts)

        t_warped = cv2.warpAffine(cropped, M, (w2, h2))

        t_warped = cv2.bitwise_and(t_warped, mask2)

        triangle_area = new_face[y2: y2 + h2, x2: x2 + w2]

        triangle_area_gray = cv2.cvtColor(triangle_area, cv2.COLOR_BGR2GRAY)

        _, triangle_mask = cv2.threshold(triangle_area_gray, 1, 255, cv2.THRESH_BINARY_INV)

        t_warped = cv2.bitwise_and(t_warped, t_warped, mask=triangle_mask)

        triangle_area = cv2.add(triangle_area, t_warped)

        new_face[y2: y2 + h2, x2: x2 + w2] = triangle_area

    new_face_gray = cv2.cvtColor(new_face, cv2.COLOR_BGR2GRAY)

    _, mask2 = cv2.threshold(new_face_gray, 1, 255, cv2.THRESH_BINARY_INV)

    ret, background = cv2.threshold(new_face_gray, 1, 255, cv2.THRESH_BINARY)

    no_face = cv2.bitwise_and(img2, img2, mask=mask2)

    resulting = cv2.add(new_face, no_face)

    seamlessclone = cv2.seamlessClone(resulting, img2, background, center2, cv2.NORMAL_CLONE)

    return seamlessclone


if __name__ == "__main__":
    photos = glob("photos/*")

    for photo in photos:
        swapped_face = photo.split('/')[-1]

        img = cv2.imread(f"photos/{swapped_face}")

        faces = detector(img)

        landmark_pts = []

        triangle_index = []

        for face in faces:
            landmark = predictor(img, face)

            for n in range(0, 68):
                x = landmark.part(n).x
                y = landmark.part(n).y

                landmark_pts.append((x, y))

            pts = np.array(landmark_pts)

            hull = cv2.convexHull(pts)
            # cv2.polylines(img, [hull], True, (255, 0, 0), 3)

            mask = np.zeros_like(img)

            cv2.fillConvexPoly(mask, hull, (255, 255, 255))

            final_img = cv2.bitwise_and(img, mask)

            rect = cv2.boundingRect(hull)

            subdiv = cv2.Subdiv2D(rect)

            subdiv.insert(landmark_pts)

            triangles = subdiv.getTriangleList()
            triangles = np.array(triangles, dtype= int)

            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])

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
                    triangle_index.append((ind1, ind2, ind3))

            # cv2.imshow("Delaunay Triangulation", img)

        videos = glob("videos/*")

        for video in videos:
            src = video.split('/')[-1].split('.')[0]

            cap = cv2.VideoCapture(f"videos/{src}.mp4")
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            size = (int(w), int(h))

            result_path = "trash"

            if not os.path.exists(result_path):
                os.makedirs(result_path)

            path = os.path.join(result_path, f'{swapped_face.split(".")[0]}_{src}.avi')

            out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), cap.get(cv2.CAP_PROP_FPS), size)

            while True:
                ret, frame = cap.read()

                if not ret:
                    cap.release()
                    break

                result = main(img, frame, landmark_pts, triangle_index)

                out.write(result)

            out.release()

            clip = VideoFileClip(f'trash/{swapped_face.split(".")[0]}_{src}.avi')

            clip2 = VideoFileClip(f'videos/{src}.mp4')

            clip.audio = clip2.audio

            if not os.path.exists('result'):
                os.makedirs('result')

            clip.write_videofile(f'result/{src}_{swapped_face.split(".")[0]}.mp4')
