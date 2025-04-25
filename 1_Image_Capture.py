import cv2
import os

# Prompt for the main directory (e.g., train or test)
main_dir = input("Enter the main directory name (e.g., train/test): ")
if not os.path.exists(main_dir):
    os.makedirs(main_dir)

cap = cv2.VideoCapture(0)
print("Press 'c' to start capturing 2000 images for a class.")
print("Press Ctrl+C in terminal to stop.")

try:
    while True:
        # Ask for the class/subdirectory name
        class_name = input("\nEnter subdirectory name for the next class (e.g., A, B, 1, 2): ")
        class_path = os.path.join(main_dir, class_name)

        # Create subdirectory if it doesn't exist
        if not os.path.exists(class_path):
            os.makedirs(class_path)
            print(f"Created directory: {class_path}")
        else:
            print(f"Directory already exists: {class_path}")

        print("Press 'c' to start capturing images...")

        capturing = False
        count = 0
        max_images = 500

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            x1, y1, x2, y2 = 300, 10, 620, 310
            roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
            _, test_image = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            test_image = cv2.resize(test_image, (310, 310))

            # Show frames
            cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255, 0, 0), 1)
            cv2.putText(frame, f"Capturing: {count}/{max_images}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Main Frame", frame)
            cv2.imshow("Processed ROI", test_image)

            key = cv2.waitKey(1)

            if not capturing and key & 0xFF == ord('c'):
                capturing = True
                print("Started capturing...")

            if capturing:
                cv2.imwrite(os.path.join(class_path, f"{count}.jpg"), test_image)
                count += 1
                if count >= max_images:
                    print(f"Finished capturing 2000 images for class '{class_name}'.")
                    break

except KeyboardInterrupt:
    print("\nProgram terminated by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
