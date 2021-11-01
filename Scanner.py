import cv2
import numpy as np
import utlis

webCamFeed = True
path_image = "1.jpg"
capture = cv2.VideoCapture(0)
capture.set(10, 160)
height_img = 640
width_img = 480
########################################################################

utlis.initializeTrackbars()
count = 0

while True:

    if webCamFeed:
        success, img = capture.read()
    else:
        img = cv2.imread(path_image)
    img = cv2.resize(img, (width_img, height_img))  # RESIZE IMAGE
    imgBlank = np.zeros((height_img, width_img, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    threshold = utlis.valTrackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
    img_Threshold = cv2.Canny(imgBlur, threshold[0], threshold[1]) # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(img_Threshold, kernel, iterations=2)  # APPLY DILATION
    img_Threshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

    ## FIND ALL CONTOURS
    img_Contours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(img_Threshold, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    cv2.drawContours(img_Contours, contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS

    # FIND THE BIGGEST CONTOUR
    biggest, maxArea = utlis.biggestContour(contours)
    if biggest.size != 0:
        biggest = utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
        imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp_colored = cv2.warpPerspective(img, matrix, (width_img, height_img))

        # REMOVE 20 PIXELS FORM EACH SIDE
        img_warp_colored = img_warp_colored [20:img_warp_colored.shape[0] - 20, 20:img_warp_colored.shape[1] - 20]
        img_warp_colored = cv2.resize(img_warp_colored, (width_img, height_img))

        # APPLY ADAPTIVE THRESHOLD
        img_warp_gray = cv2.cvtColor(img_warp_colored, cv2.COLOR_BGR2GRAY)
        img_adaptive_threshold = cv2.adaptiveThreshold(img_warp_gray, 255, 1, 1, 7, 2)
        img_adaptive_threshold = cv2.bitwise_not(img_adaptive_threshold)
        img_adaptive_threshold = cv2.medianBlur(img_adaptive_threshold, 3)

        # Image Array for Display
        image_array = ([img, imgGray, img_Threshold, img_Contours],
                       [imgBigContour, img_warp_colored, img_warp_gray, img_adaptive_threshold])

    else:
        image_array = ([img, imgGray, img_Threshold, img_Contours],
                       [imgBlank, imgBlank, imgBlank, imgBlank])

    labels =[["Original", "Gray", "Threshold", "Contours"],
             ["Biggest Contour","Warp Perspective","Warp Gray","Adaptive Threshold"]]
    stacked_images = utlis.stackImages(image_array, 0.75, labels)
    cv2.imshow("Result", stacked_images)

    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage" + str(count) + ".jpg", img_warp_colored)
        cv2.rectangle(stacked_images, ((int(stacked_images.shape[1] / 2) - 230), int(stacked_images.shape[0] / 2) + 50), (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stacked_images, "Scan Saved", (int(stacked_images.shape[1] / 2) - 200, int(stacked_images.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stacked_images)
        cv2.waitKey(300)
        count += 1