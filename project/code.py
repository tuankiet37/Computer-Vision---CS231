import cv2
import dlib
import numpy as np

face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmark_point(img_gray, faces):
    face_landmarks = face_predictor(img_gray, faces[0])
    face_landmarks_point = []
    for landmark in range (0,68):
        x_point = face_landmarks.part(landmark).x
        y_point = face_landmarks.part(landmark).y
        face_landmarks_point.append((x_point,y_point))
    return face_landmarks_point

def find_triangle(face_landmarks_point, img, canvas):
    face_landmarks_point_array = np.array(face_landmarks_point,np.int32)
    face_convexhull = cv2.convexHull(face_landmarks_point_array) #return points which lie on the boundary of face
    #cv2.fillConvexPoly(canvas, face_convexhull,255) # fill in the convex poly which is created by above points
    #face_img = cv2.bitwise_and(img, img, mask = canvas)
    bounding_rectangle = cv2.boundingRect(face_convexhull) # return start point, end point, height, width
    subdivsions = cv2.Subdiv2D(bounding_rectangle) #create an empty Delaunay subdivison
    subdivsions.insert(face_landmarks_point) #insert the face landmark points into subdivisions
    triangles_vector = subdivsions.getTriangleList() # return list points of Delaunany triangles
    triangles_array = np.array(triangles_vector, dtype=np.int32)

    triangle_index_point_list = [] 
    for triangle in triangles_array:
        index_point1 = (triangle[0], triangle[1])
        index_point2 = (triangle[2], triangle[3])
        index_point3 = (triangle[4], triangle[5])
        #find index cua 3 diem di voi nhau
        index_point1 = np.where((face_landmarks_point_array == index_point1).all(axis=1))[0][0] 
        index_point2 = np.where((face_landmarks_point_array == index_point2).all(axis=1))[0][0]
        index_point3 = np.where((face_landmarks_point_array == index_point3).all(axis=1))[0][0]

        triangle=[index_point1,index_point2,index_point3]
        triangle_index_point_list.append(triangle)    
    return triangle_index_point_list, face_convexhull

def wrap_triangle (origin_triangle_index_point_list, origin_face_landmarks_point, origin_img, target_triangle_index_point_list, target_face_landmarks_point, target_img, target_canvas):
    for triangle_index_points in (origin_triangle_index_point_list):
        origin_triangle_point1 = origin_face_landmarks_point[triangle_index_points[0]]
        origin_triangle_point2 = origin_face_landmarks_point[triangle_index_points[1]]
        origin_triangle_point3 = origin_face_landmarks_point[triangle_index_points[2]]
        origin_triangle = np.array([origin_triangle_point1,origin_triangle_point2,origin_triangle_point3], np.int32)
        bounding_rectangle = cv2.boundingRect(origin_triangle)
        (x,y,w,h) = bounding_rectangle
        origin_cropped_rectangle = origin_img[y:y+h, x:x+w]
        origin_cropped_rectangle_mask = np.zeros((h, w), np.uint8)
        origin_triangle_points = np.array([[origin_triangle_point1[0]-x, origin_triangle_point1[1]-y],
                                [origin_triangle_point2[0]-x, origin_triangle_point2[1]-y],
                                [origin_triangle_point3[0]-x, origin_triangle_point3[1]-y]], dtype=np.uint8)
        #cv2.fillConvexPoly(origin_cropped_rectangle_mask, origin_triangle_points, 255)


        target_triangle_point1 = target_face_landmarks_point[triangle_index_points[0]]
        target_triangle_point2 = target_face_landmarks_point[triangle_index_points[1]]
        target_triangle_point3 = target_face_landmarks_point[triangle_index_points[2]]
        target_triangle = np.array([target_triangle_point1,target_triangle_point2,target_triangle_point3], np.int32)
        bounding_rectangle = cv2.boundingRect(target_triangle)
        (x,y,w,h) = bounding_rectangle
        target_cropped_rectangle = target_img[y:y+h, x:x+w]
        target_cropped_rectangle_mask = np.zeros((h, w), np.uint8)
        target_triangle_points = np.array([[target_triangle_point1[0]-x, target_triangle_point1[1]-y],
                                [target_triangle_point2[0]-x, target_triangle_point2[1]-y],
                                [target_triangle_point3[0]-x, target_triangle_point3[1]-y]], dtype=np.int32)
        cv2.fillConvexPoly(target_cropped_rectangle_mask, target_triangle_points, 255)
        
        
        origin_triangle_points = np.float32(origin_triangle_points)
        target_triangle_points = np.float32(target_triangle_points)

        
        M = cv2.getAffineTransform(origin_triangle_points,target_triangle_points)
        warped_triangle = cv2.warpAffine(origin_cropped_rectangle, M, (w,h))
       
       
       
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask= target_cropped_rectangle_mask)
        #cv2.imshow("Before", warped_triangle)
        # cv2.waitKey(0)

        new_target_face_canvas = target_canvas [y: y+h, x: x+w]
        # cv2.imshow("T", new_target_face_canvas)
        new_target_face_canvas_gray = cv2.cvtColor(new_target_face_canvas, cv2.COLOR_BGR2GRAY)

        _, mask_created_triangle =  cv2.threshold (new_target_face_canvas_gray, 1, 255, cv2.THRESH_BINARY_INV)
        #cv2.imshow("Threshold", mask_created_triangle)
        # cv2.waitKey(0)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask = mask_created_triangle)
        #place the masked triangle inside the small canvas area
        new_target_face_canvas = cv2.add(new_target_face_canvas, warped_triangle)
        #place the new small canvas with triangle in it to the large destination canvas
        #at the designated location
        target_canvas[y: y+h, x: x+w] = new_target_face_canvas
        # cv2.imshow("Wraped triangle", warped_triangle)
        # cv2.waitKey(0)
    return target_canvas



#Origin_img
origin_img = cv2.imread("jason.jpg")
origin_img_gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
origin_canvas = np.zeros_like (origin_img_gray)
origin_faces = face_detector(origin_img)
origin_face_landmarks_point = get_landmark_point(origin_img_gray,origin_faces)
origin_triangle_index_point_list,_ = find_triangle(origin_face_landmarks_point, origin_img, origin_canvas)


#Target_img
target_img = cv2.imread("brucewills.jpg")
target_img_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
target_canvas = np.zeros_like (target_img)
target_faces = face_detector(target_img)
target_face_landmarks_point = get_landmark_point(target_img_gray,target_faces)
target_triangle_index_point_list,target_face_convexhull = find_triangle(target_face_landmarks_point, target_img, target_canvas)


target_image_canvas = wrap_triangle (origin_triangle_index_point_list, origin_face_landmarks_point, origin_img, target_triangle_index_point_list, target_face_landmarks_point, target_img, target_canvas)
final_target_canvas = np.zeros_like(target_img_gray)
cv2.imshow("b", target_image_canvas)  
    
#create the target face mask
final_target_face_mask = cv2.fillConvexPoly(final_target_canvas, target_face_convexhull, 255)
cv2.imshow("T", final_target_face_mask) 
    
#invert the face mask color
final_target_canvas = cv2.bitwise_not(final_target_face_mask) 
cv2.imshow("a", final_target_canvas)         
#mask target face
target_face_masked = cv2.bitwise_and(target_img, target_img, mask=final_target_canvas)  

target_with_face = cv2.add(target_face_masked,target_image_canvas)  
cv2.imshow("1", target_with_face)      
(x,y,w,h) = cv2.boundingRect(target_face_convexhull)
 
point = ((x+x+w)//2, (y+y+h)//2)
target_with_face = 	cv2.seamlessClone(target_with_face, target_img, final_target_face_mask, point, cv2.NORMAL_CLONE) 
cv2.imshow("Final", target_with_face) 
cv2.waitKey(0)