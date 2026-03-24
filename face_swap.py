"""
Face swap script: Replaces a face in a group photo with a face from a selfie.

Usage:
    python face_swap.py selfie.jpg group_photo.jpg output.jpg

Requirements:
    pip install Pillow numpy opencv-python
"""

import sys
import cv2
import numpy as np
from PIL import Image


def detect_faces(image, scale_factor=1.1, min_neighbors=5):
    """Detect faces using OpenCV's Haar cascade."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(50, 50)
    )
    return faces


def get_largest_face(faces):
    """Return the largest detected face (most likely the main subject)."""
    if len(faces) == 0:
        return None
    areas = [w * h for (x, y, w, h) in faces]
    return faces[np.argmax(areas)]


def swap_face(selfie_path, group_path, output_path, target_face_index=None):
    """
    Extract face from selfie and blend it onto a face in the group photo.

    Args:
        selfie_path: Path to the selfie image (source face)
        group_path: Path to the group photo (destination)
        output_path: Path for the output image
        target_face_index: Which face in the group to replace (0-indexed by size).
                          None = largest face. Use --list to see detected faces.
    """
    # Load images
    selfie = cv2.imread(selfie_path)
    group = cv2.imread(group_path)

    if selfie is None:
        print(f"Error: Could not load selfie from {selfie_path}")
        sys.exit(1)
    if group is None:
        print(f"Error: Could not load group photo from {group_path}")
        sys.exit(1)

    # Detect faces
    selfie_faces = detect_faces(selfie, scale_factor=1.05, min_neighbors=3)
    group_faces = detect_faces(group, scale_factor=1.05, min_neighbors=3)

    print(f"Detected {len(selfie_faces)} face(s) in selfie")
    print(f"Detected {len(group_faces)} face(s) in group photo")

    if len(selfie_faces) == 0:
        print("No face detected in selfie. Try a clearer photo.")
        sys.exit(1)
    if len(group_faces) == 0:
        print("No faces detected in group photo.")
        sys.exit(1)

    # Get source face (largest in selfie)
    src_face = get_largest_face(selfie_faces)
    sx, sy, sw, sh = src_face

    # Add padding around the detected face for more natural look
    pad = int(0.3 * max(sw, sh))
    sx_p = max(0, sx - pad)
    sy_p = max(0, sy - pad)
    sw_p = min(selfie.shape[1] - sx_p, sw + 2 * pad)
    sh_p = min(selfie.shape[0] - sy_p, sh + 2 * pad)

    face_region = selfie[sy_p : sy_p + sh_p, sx_p : sx_p + sw_p]

    # Select target face in group photo
    if target_face_index is not None:
        # Sort by area descending
        sorted_faces = sorted(group_faces, key=lambda f: f[2] * f[3], reverse=True)
        if target_face_index >= len(sorted_faces):
            print(f"Only {len(sorted_faces)} faces found. Index {target_face_index} out of range.")
            sys.exit(1)
        dst_face = sorted_faces[target_face_index]
    else:
        dst_face = get_largest_face(group_faces)

    dx, dy, dw, dh = dst_face

    # Add padding to destination too
    d_pad = int(0.3 * max(dw, dh))
    dx_p = max(0, dx - d_pad)
    dy_p = max(0, dy - d_pad)
    dw_p = min(group.shape[1] - dx_p, dw + 2 * d_pad)
    dh_p = min(group.shape[0] - dy_p, dh + 2 * d_pad)

    # Resize source face to match destination face size
    resized_face = cv2.resize(face_region, (dw_p, dh_p), interpolation=cv2.INTER_LANCZOS4)

    # Create an elliptical mask for smooth blending
    mask = np.zeros((dh_p, dw_p), dtype=np.float32)
    center = (dw_p // 2, dh_p // 2)
    axes = (int(dw_p * 0.42), int(dh_p * 0.45))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

    # Feather the mask edges for smooth blending
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=dw_p * 0.08, sigmaY=dh_p * 0.08)

    # Color-match the source face to the destination region
    dst_region = group[dy_p : dy_p + dh_p, dx_p : dx_p + dw_p]

    # Simple color transfer (match mean and std of each channel)
    src_lab = cv2.cvtColor(resized_face, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(dst_region, cv2.COLOR_BGR2LAB).astype(np.float32)

    for i in range(3):
        src_mean, src_std = src_lab[:, :, i].mean(), src_lab[:, :, i].std()
        dst_mean, dst_std = dst_lab[:, :, i].mean(), dst_lab[:, :, i].std()
        if src_std > 0:
            src_lab[:, :, i] = (src_lab[:, :, i] - src_mean) * (dst_std / src_std) + dst_mean

    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    color_matched = cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)

    # Blend using the mask
    result = group.copy()
    mask_3ch = np.stack([mask] * 3, axis=-1)
    blended = (color_matched * mask_3ch + dst_region * (1 - mask_3ch)).astype(np.uint8)
    result[dy_p : dy_p + dh_p, dx_p : dx_p + dw_p] = blended

    # Save result
    cv2.imwrite(output_path, result)
    print(f"Face swap complete! Saved to {output_path}")

    # Also show which faces were detected (for --list mode)
    return group_faces


def list_faces(group_path):
    """Show detected faces with indices for selection."""
    group = cv2.imread(group_path)
    if group is None:
        print(f"Error: Could not load {group_path}")
        sys.exit(1)

    faces = detect_faces(group, scale_factor=1.05, min_neighbors=3)
    sorted_faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    print(f"\nDetected {len(sorted_faces)} face(s):")
    for i, (x, y, w, h) in enumerate(sorted_faces):
        print(f"  [{i}] position=({x},{y}) size={w}x{h}")

    # Save annotated image
    annotated = group.copy()
    for i, (x, y, w, h) in enumerate(sorted_faces):
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(annotated, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    out = group_path.rsplit(".", 1)
    annotated_path = f"{out[0]}_faces.{out[1]}"
    cv2.imwrite(annotated_path, annotated)
    print(f"Annotated image saved to {annotated_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nExamples:")
        print("  python face_swap.py selfie.jpg group.jpg output.jpg")
        print("  python face_swap.py --list group.jpg")
        print("  python face_swap.py selfie.jpg group.jpg output.jpg --target 2")
        sys.exit(0)

    if sys.argv[1] == "--list":
        list_faces(sys.argv[2])
    else:
        selfie_path = sys.argv[1]
        group_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else "output.jpg"

        target_idx = None
        if "--target" in sys.argv:
            ti = sys.argv.index("--target")
            target_idx = int(sys.argv[ti + 1])

        swap_face(selfie_path, group_path, output_path, target_idx)
