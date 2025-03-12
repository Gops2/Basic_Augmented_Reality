#Using python(4.6.0.66). To install it: pip install opencv-contrib-python==4.6.0.66
import cv2
import numpy as np



#Camera Calibration Matrix and Camera Distortion Coefficients, from Q3.
K = np.array([[1.07537472e+03, 0.00000000e+00, 6.43248864e+02],                 
              [0.00000000e+00, 1.07531177e+03, 3.27424833e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeffs = np.array([0.04922674535879292, 0.26400257262435856, -0.006248237566977403, 6.221352287141695e-05, -1.3229381574999024])


image = cv2.imread('AR6.jpg')                                       #Upload image, File names:AR4,AR5,AR6

if image is None:
    print("Image not found. Please check the file path.")
else:
    #aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_parameters = cv2.aruco.DetectorParameters_create()

    # Detect the ArUco markers in the image
    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_parameters)

    if ids is not None:                                                                                                 
        for i in range(len(ids)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, K, dist_coeffs)

            axis_length = 0.02                                                                                          #Define Axes
            axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(-1, 3)

            image_points, _ = cv2.projectPoints(axis_points, rvec, tvec, K, dist_coeffs)
            image_points = image_points.astype(int)

    
            cv2.line(image, tuple(image_points[0].ravel()), tuple(image_points[1].ravel()), (0, 0, 255), 3)             #X-axis (red)
            cv2.line(image, tuple(image_points[0].ravel()), tuple(image_points[2].ravel()), (0, 255, 0), 3)             #Y-axis (green)
            cv2.line(image, tuple(image_points[0].ravel()), tuple(image_points[3].ravel()), (255, 0, 0), 3)             #Z-axis (blue)
            cv2.aruco.drawDetectedMarkers(image, corners)                                                               #To draw cube on ARUCO marker

           
            cube_points = np.array([[-0.01, -0.01, 0], [-0.01, 0.01, 0], [0.01, 0.01, 0], [0.01, -0.01, 0],             #Define the 3D cube points in the marker coordinate system
                        [-0.01, -0.01, 0.02], [-0.01, 0.01, 0.02], [0.01, 0.01, 0.02], [0.01, -0.01, 0.02]])

            
            image_points, _ = cv2.projectPoints(cube_points, rvec, tvec, K, dist_coeffs)                                #Project the 3D cube points to the 2D image
            image_points = image_points.astype(int)

            for j in range(4):                                                                                          #Connect the cube points to draw the cube
                cv2.line(image, tuple(image_points[j].ravel()), tuple(image_points[j + 4].ravel()), (0, 0, 255), 2)  
                cv2.line(image, tuple(image_points[j].ravel()), tuple(image_points[(j + 1) % 4].ravel()), (0, 0, 255), 2)  
                cv2.line(image, tuple(image_points[j + 4].ravel()), tuple(image_points[(j + 1) % 4 + 4].ravel()), (0, 0, 255), 2)  
    else:
        print("Retry. Ids is empty.")
    

    #Visualisation
    cv2.imshow('AR Cube', image)
    cv2.imwrite('Q4_Solution_AR6.png', image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
