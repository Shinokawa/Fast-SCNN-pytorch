import cv2
import numpy as np
import os
import argparse

# Global variables
points = []
current_image = None
mask = None
window_name = 'Create Mask'

def mouse_callback(event, x, y, flags, param):
    global points, current_image, mask

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(current_image, (x, y), 3, (0, 255, 0), -1)
        if len(points) > 1:
            cv2.line(current_image, points[-2], points[-1], (0, 255, 0), 2)
        cv2.imshow(window_name, current_image)

def create_mask_for_image(image_path, output_dir):
    global points, current_image, mask
    current_image = cv2.imread(image_path)
    if current_image is None:
        print(f"Error: Could not load image {image_path}")
        return

    clone = current_image.copy()
    mask = np.zeros((current_image.shape[0], current_image.shape[1]), dtype=np.uint8)
    points = []

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\nInstructions:")
    print(" - Left-click to draw the polygon around the lane.")
    print(" - Press 'f' to fill the current polygon (creates the mask for one lane).")
    print(" - After filling, you can start drawing a new polygon for the other lane.")
    print(" - Press 's' to save the final mask and move to the next image.")
    print(" - Press 'r' to reset all polygons on the current image.")
    print(" - Press 'q' to quit.")

    while True:
        cv2.imshow(window_name, current_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Resetting points.")
            points = []
            current_image = clone.copy()
            mask = np.zeros((current_image.shape[0], current_image.shape[1]), dtype=np.uint8)
            cv2.imshow(window_name, current_image)
        elif key == ord('f'):
            if len(points) > 2:
                print("Filling polygon.")
                cv2.fillPoly(mask, [np.array(points)], 1) # Fill with 1 for the lane
                # Also draw on the display image to show the filled area
                cv2.fillPoly(current_image, [np.array(points)], (0, 100, 0, 0.5)) 
                points = [] # Reset points for the next lane
            else:
                print("You need at least 3 points to form a polygon.")
        elif key == ord('s'):
            output_filename = os.path.basename(image_path).replace('.jpg', '.png').replace('.jpeg', '.png')
            output_path = os.path.join(output_dir, output_filename)
            # The mask is already 0 for background and 1 for lane.
            # We need to save it as a single channel image.
            # The saved png will have values 0 and 1, which is what we need.
            cv2.imwrite(output_path, mask)
            print(f"Mask saved to {output_path}")
            break

    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Create segmentation masks for lane detection.")
    parser.add_argument('--input-dir', type=str, default='data/custom/images',
                        help='Directory containing the images to be annotated.')
    parser.add_argument('--output-dir', type=str, default='data/custom/masks',
                        help='Directory where the masks will be saved.')
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Input directory '{args.input_dir}' not found. Please create it and add your images.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(args.input_dir, image_file)
        print(f"\nProcessing image: {image_path}")
        create_mask_for_image(image_path, args.output_dir)

if __name__ == '__main__':
    main()
