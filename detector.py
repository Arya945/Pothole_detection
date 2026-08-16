import cv2
import numpy as np

class detector():
    def __init__(self,cfg,weights,names):
        self.cfg_path = cfg
        self.weights_path = weights
        self.names_path = names

        self.classes = ["potholes"]

        self.net = cv2.dnn.readNetFromDarknet(self.cfg_path, self.weights_path)

    def detect(self, img, conf_threshold):
        self.image_path = img
        self.img = cv2.imread(self.image_path)
        (H, W) = self.img.shape[:2]

        # Prepare input
        blob = cv2.dnn.blobFromImage(
            self.img, 1/255.0, (416, 416),
            swapRB=True, crop=False
        )
        self.net.setInput(blob)

        # Get output layer names
        ln = self.net.getUnconnectedOutLayersNames()

        # Run forward pass
        layer_outputs = self.net.forward(ln)

        # Process detections
        nms_threshold = 0.3

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    # Scale bounding box to original image size
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maxima Suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences,
            score_threshold=conf_threshold,
            nms_threshold=nms_threshold
        )

        # Draw boxes
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = (0, 255, 0)  # green box
                cv2.rectangle(self.img, (x, y), (x + w, y + h), color, 2)

                text = f"{self.classes[class_ids[i]]}: {confidences[i]:.2f}"
                cv2.putText(self.img, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        cv2.imwrite("output.jpg", self.img)

    def show(self):    
        cv2.imshow("YOLOv4 Detection", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ =='__main__':
    cfg_path = r"files\pothole.cfg"
    weights_path = r"files\yolov4.weights"
    names_path = r"files\data.names"

    det = detector(cfg_path,weights_path,names_path)

    det.detect("R.jpeg", 0.3)
    det.show()