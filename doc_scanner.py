from imutils.perspective import four_point_transform, order_points
from skimage.filters import threshold_local
import argparse
import cv2
import imutils
import numpy as np

# Working height used for edge/contour detection. Higher than before for
# more precise geometry; the final warp is computed at full resolution.
WORKING_HEIGHT = 1000

# Canny threshold pairs tried in order; combined with a morphological
# gradient pass to build up a robust set of edge candidates.
CANNY_THRESHOLDS = [(50, 150), (75, 150), (100, 200), (30, 100)]

MIN_CONTOUR_FRACTION = 0.02

SCAN_MODES = ("bw", "gray", "color")


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"-i",
		"--image",
		required=True,
		help="Path to the image to be scanned",
	)
	parser.add_argument(
		"-m",
		"--mode",
		choices=SCAN_MODES,
		default="bw",
		help="Output mode for the scan (default: bw)",
	)
	return vars(parser.parse_args())


def load_and_resize_image(image_path):
	image = cv2.imread(image_path)
	if image is None:
		raise ValueError(f"Could not read image from path: {image_path}")

	original = image.copy()
	resized = imutils.resize(image, height=WORKING_HEIGHT)
	ratio = original.shape[0] / float(resized.shape[0])
	return original, resized, ratio


def edge_candidates(gray):
	"""Return several edge maps tuned differently for robust detection."""
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	maps = [cv2.Canny(blurred, low, high) for low, high in CANNY_THRESHOLDS]

	gradient = cv2.morphologyEx(
		gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	)
	gradient = cv2.GaussianBlur(gradient, (3, 3), 0)
	_, gradient = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	maps.append(gradient)
	return maps


def _collect_quads(edge_maps, height, width):
	"""Collect deduplicated, convex 4-point approximations from all edge maps."""
	image_area = float(height * width)
	min_area = MIN_CONTOUR_FRACTION * image_area
	margin = max(4.0, 0.03 * min(height, width))
	seen = np.empty((0, 2))
	quads = []

	for edged in edge_maps:
		contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		contours = imutils.grab_contours(contours)

		for contour in contours:
			area = cv2.contourArea(contour)
			if area < min_area:
				continue

			perimeter = cv2.arcLength(contour, True)
			if perimeter <= 0:
				continue

			approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
			if len(approx) != 4 or not cv2.isContourConvex(approx):
				continue

			points = approx.reshape(4, 2)
			center = np.array([points[:, 0].mean(), points[:, 1].mean()])
			if seen.size and np.any(np.abs(seen - center).max(axis=1) < margin):
				continue

			seen = center[None, :] if not seen.size else np.vstack((seen, center[None, :]))
			quads.append((area, approx))

	return quads


def quad_score(contour, height, width):
	"""Score a candidate rectangle: prefer large, document-shaped, interior quads."""
	area = cv2.contourArea(contour)
	image_area = float(height * width)
	if area <= 0:
		return -1.0

	normalized_area = min(area / image_area, 1.0)
	if normalized_area < MIN_CONTOUR_FRACTION:
		return -1.0

	points = order_points(contour.reshape(4, 2).astype("float32"))
	top = np.linalg.norm(points[0] - points[1])
	right = np.linalg.norm(points[1] - points[2])
	bottom = np.linalg.norm(points[2] - points[3])
	left = np.linalg.norm(points[3] - points[0])
	width_len = max(top, bottom)
	height_len = max(right, left)
	aspect = max(width_len, height_len) / max(min(width_len, height_len), 1.0)

	aspect_score = 1.0 if aspect <= 2.5 else max(0.0, 1.0 - (aspect - 2.5) * 0.5)

	margin_px = 0.03 * min(height, width)
	near_frame = sum(
		1
		for x, y in points
		if x < margin_px or x > width - margin_px or y < margin_px or y > height - margin_px
	)
	frame_score = 1.0 - 0.12 * near_frame

	return 0.5 * normalized_area + 0.3 * aspect_score + 0.2 * frame_score


def find_document_contour(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edge_maps = edge_candidates(gray)
	height, width = gray.shape[:2]
	quads = _collect_quads(edge_maps, height, width)

	best, best_score = None, -1.0
	for area, contour in quads:
		score = quad_score(contour, height, width)
		if score > best_score:
			best, best_score = contour, score

	return best, edge_maps[0]


def enhance_gray(gray):
	h, w = gray.shape[:2]
	kernel_size = max(21, (min(h, w) // 8) | 1)

	background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
	corrected = cv2.divide(gray, background, scale=255)
	corrected = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(corrected)
	corrected = cv2.fastNlMeansDenoising(
		corrected, None, h=3, templateWindowSize=7, searchWindowSize=21
	)

	blurred = cv2.GaussianBlur(corrected, (0, 0), 2.0)
	return cv2.addWeighted(corrected, 1.8, blurred, -0.8, 0)


def render_scan(warped, mode="bw"):
	h, w = warped.shape[:2]
	kernel_size = max(21, (min(h, w) // 8) | 1)

	if mode == "color":
		background = [
			cv2.GaussianBlur(warped[:, :, i], (kernel_size, kernel_size), 0) for i in range(3)
		]
		out = np.empty_like(warped)
		for i in range(3):
			out[:, :, i] = cv2.divide(warped[:, :, i], background[i], scale=255)

		lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
		lightness, a, b = cv2.split(lab)
		lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
		out = cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)

		blurred = cv2.GaussianBlur(out, (0, 0), 2.0)
		return cv2.addWeighted(out, 1.8, blurred, -0.8, 0)

	gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	enhanced = enhance_gray(gray)

	if mode == "gray":
		return enhanced

	threshold = threshold_local(enhanced, 11, offset=8, method="gaussian")
	return (enhanced > threshold).astype("uint8") * 255


def perspective_scan(original_image, contour, ratio, mode="bw"):
	warped = four_point_transform(original_image, contour.reshape(4, 2) * ratio)
	return render_scan(warped, mode)


def scan_document_image(image, mode="bw", corners=None):
	if image is None:
		raise ValueError("No image data provided")
	if mode not in SCAN_MODES:
		raise ValueError(f"Unknown scan mode {mode!r}. Choose from {SCAN_MODES}.")

	original = image.copy()
	ratio = original.shape[0] / float(WORKING_HEIGHT)
	resized = imutils.resize(image, height=WORKING_HEIGHT)
	edged = None

	if corners is not None:
		corners_array = np.array(corners, dtype="float32").reshape(4, 2)
		document_contour = corners_array / ratio
	else:
		document_contour, edged = find_document_contour(resized)
		if document_contour is None:
			raise ValueError("Could not find a 4-point document contour in the image")

	scanned = perspective_scan(original, document_contour, ratio, mode)
	return original, scanned, edged, document_contour


def show_outline(image, contour):
	outlined = image.copy()
	cv2.drawContours(outlined, [contour], -1, (0, 255, 0), 2)
	cv2.imshow("Outline", outlined)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def show_result(original, scanned):
	cv2.imshow("Original", imutils.resize(original, height=500))
	cv2.imshow("Scanned", imutils.resize(scanned, height=500))
	cv2.waitKey(0)


def main():
	args = parse_arguments()
	original, resized, ratio = load_and_resize_image(args["image"])
	document_contour, _ = find_document_contour(resized)

	if document_contour is None:
		raise ValueError("Could not find a 4-point document contour in the image")

	show_outline(resized, document_contour)
	scanned = perspective_scan(original, document_contour, ratio, args["mode"])
	show_result(original, scanned)


if __name__ == "__main__":
	main()