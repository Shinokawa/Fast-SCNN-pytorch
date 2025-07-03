import cv2
import numpy as np
import os
import argparse

# Global variables
points = []
current_image = None
mask = None
window_name = 'Create Mask'
polygons_history = []  # Store completed polygons for undo
preview_mode = False
brush_size = 8
drawing = False
last_point = None
display_scale = 1.5  # Scale factor for display

def redraw_current_image():
    global points, current_image, clone, display_scale
    current_image = clone.copy()
    # Redraw all points and lines
    for i, point in enumerate(points):
        cv2.circle(current_image, point, 3, (0, 255, 0), -1)
        if i > 0:
            cv2.line(current_image, points[i-1], points[i], (0, 255, 0), 2)
    
    # Scale up for display
    display_img = cv2.resize(current_image, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(window_name, display_img)

def show_progress(current_idx, total_images):
    print(f"\nProgress: {current_idx + 1}/{total_images} images")
    remaining = total_images - current_idx - 1
    if remaining > 0:
        print(f"Remaining: {remaining} images")

def mouse_callback(event, x, y, flags, param):
    global points, current_image, mask, preview_mode, drawing, last_point, brush_size, display_scale

    # Convert display coordinates back to original image coordinates
    orig_x = int(x / display_scale)
    orig_y = int(y / display_scale)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_point = (orig_x, orig_y)
        points.append((orig_x, orig_y))
        cv2.circle(current_image, (orig_x, orig_y), 3, (0, 255, 0), -1)
        
        # Scale up for display
        display_img = cv2.resize(current_image, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(window_name, display_img)
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and (flags & cv2.EVENT_FLAG_LBUTTON):
            # Continuous drawing mode
            if last_point:
                # Draw line from last point to current point
                cv2.line(current_image, last_point, (orig_x, orig_y), (0, 255, 0), 2)
                cv2.line(mask, last_point, (orig_x, orig_y), 1, brush_size)
                
                # Add intermediate points for smooth curves
                distance = np.sqrt((orig_x - last_point[0])**2 + (orig_y - last_point[1])**2)
                if distance > 5:  # Only add point if moved enough
                    points.append((orig_x, orig_y))
                
                last_point = (orig_x, orig_y)
                
                # Scale up for display
                display_img = cv2.resize(current_image, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
                cv2.imshow(window_name, display_img)
                
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_point = None
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click to remove last point
        if points:
            points.pop()
            # Redraw the image
            redraw_current_image()
            
    elif event == cv2.EVENT_MOUSEWHEEL:
        # Mouse wheel to change brush size
        if flags > 0:  # Scroll up
            brush_size = min(brush_size + 1, 20)
        else:  # Scroll down
            brush_size = max(brush_size - 1, 1)
        print(f"Brush size: {brush_size}")

def create_mask_for_image(image_path, output_dir, current_idx=0, total_images=1):
    global points, current_image, mask, clone, polygons_history, preview_mode, display_scale, drawing, brush_size
    current_image = cv2.imread(image_path)
    if current_image is None:
        print(f"Error: Could not load image {image_path}")
        return

    clone = current_image.copy()
    mask = np.zeros((current_image.shape[0], current_image.shape[1]), dtype=np.uint8)
    points = []
    polygons_history = []
    preview_mode = False
    drawing = False

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    show_progress(current_idx, total_images)
    print(f"\nAnnotating: {os.path.basename(image_path)} [Display Scale: {display_scale}x]")
    print("Instructions:")
    print(" - Left-click and drag: Draw continuous lines for lane marking")
    print(" - Right-click: Remove last point") 
    print(" - Mouse wheel: Change brush size")
    print(" - 'f': Fill current drawn area (lane area)")
    print(" - 'c': Clear current drawing")
    print(" - 's': Save mask and next image")
    print(" - 'r': Reset current image")
    print(" - 'z': Undo last polygon")
    print(" - 'p': Preview mask (toggle)")
    print(" - '+/-': Increase/decrease display size")
    print(" - 'n': Skip this image")
    print(" - 'q': Quit")
    print(" - ESC: Quick save and next")
    print(f"Current brush size: {brush_size}")

    # Initial display
    display_img = cv2.resize(current_image, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(window_name, display_img)
    print(" - ESC: Quick save and next")

    while True:
        if preview_mode:
            # Show mask overlay
            overlay = current_image.copy()
            mask_colored = np.zeros_like(current_image)
            mask_colored[:,:,1] = mask * 255  # Green channel for lanes
            cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0, overlay)
            display_img = cv2.resize(overlay, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
            cv2.imshow(window_name, display_img)
        else:
            display_img = cv2.resize(current_image, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
            cv2.imshow(window_name, display_img)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            return False  # Quit annotation
        elif key == 27:  # ESC - quick save
            output_filename = os.path.basename(image_path).replace('.jpg', '.png').replace('.jpeg', '.png')
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, mask * 255)  # Save as 0-255 values
            print(f"Quick saved: {output_path}")
            return True
        elif key == ord('n'):  # Skip this image
            print("Skipping this image.")
            return True
        elif key == ord('r'):
            print("Resetting image.")
            points = []
            polygons_history = []
            current_image = clone.copy()
            mask = np.zeros((current_image.shape[0], current_image.shape[1]), dtype=np.uint8)
            display_img = cv2.resize(current_image, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
            cv2.imshow(window_name, display_img)
        elif key == ord('p'):
            preview_mode = not preview_mode
            print(f"Preview mode: {'ON' if preview_mode else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            display_scale = min(display_scale + 0.2, 3.0)
            print(f"Display scale: {display_scale:.1f}x")
        elif key == ord('-') or key == ord('_'):
            display_scale = max(display_scale - 0.2, 0.5)
            print(f"Display scale: {display_scale:.1f}x")
        elif key == ord('z'):
            if polygons_history:
                # Undo last polygon
                polygons_history.pop()
                mask = np.zeros((current_image.shape[0], current_image.shape[1]), dtype=np.uint8)
                for poly in polygons_history:
                    cv2.fillPoly(mask, [poly], 1)
                print("Undid last polygon.")
        elif key == ord('c'):
            print("Clearing current drawing.")
            points = []
            current_image = clone.copy()
            display_img = cv2.resize(current_image, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
            cv2.imshow(window_name, display_img)
        elif key == ord('f'):
            if len(points) > 2:
                print("Filling drawn area.")
                cv2.fillPoly(mask, [np.array(points)], 1)
                polygons_history.append(np.array(points))
                cv2.fillPoly(current_image, [np.array(points)], (0, 150, 0))
                points = []
                display_img = cv2.resize(current_image, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
                cv2.imshow(window_name, display_img)
            else:
                # If no polygon points, fill the current drawn mask areas
                if np.any(mask > 0):
                    print("Filling current brush strokes.")
                else:
                    print("Nothing to fill. Draw some lines first.")
        elif key == ord('s'):
            output_filename = os.path.basename(image_path).replace('.jpg', '.png').replace('.jpeg', '.png')
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, mask * 255)  # Save as 0-255 values
            print(f"Mask saved: {output_path}")
            return True

    cv2.destroyAllWindows()
    return True

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
    
    if not image_files:
        print(f"No image files found in {args.input_dir}")
        return
        
    total_images = len(image_files)
    print(f"Found {total_images} images to annotate.")
    print("TIP: Start with representative images first!")

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(args.input_dir, image_file)
        continue_annotation = create_mask_for_image(image_path, args.output_dir, i, total_images)
        if not continue_annotation:
            print("Annotation stopped by user.")
            break
            
    print("\nAnnotation session completed!")

if __name__ == '__main__':
    main()
