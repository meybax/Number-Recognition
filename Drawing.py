import cv2
import numpy as np

drawing = False  # true if mouse is pressed
ix, iy = -1, -1


# allows user to draw on frame and save the image
def main():

    # mouse callback function
    def draw(event, x, y, flags, param):
        global ix, iy, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.line(img, (ix, iy), (x, y), (0, 0, 0), 15)
                ix = x
                iy = y

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.line(img, (ix, iy), (x, y), (0, 0, 0), 15)
            ix = x
            iy = y

    drawing = False  # true if mouse is pressed
    ix, iy = -1, -1

    img = np.full((280, 280), 255, np.uint8)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', draw)

    while 1:
        cv2.imshow('image', img)
        k = cv2.waitKey(50) & 0xFF
        if k == ord('q'):
            break

    cv2.destroyAllWindows()

    digit = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    cv2.imwrite('number.png', digit)
