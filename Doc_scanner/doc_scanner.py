from imutils.perspective import four_point_transform
from skimage.filters import threshold_local
import argparse
import cv2
import imutils


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"-i",
		"--image",
		required=True,
		help="Path to the image to be scanned",
	)
	return vars(parser.parse_args())


def load_and_resize_image(image_path):
	image = cv2.imread(image_path)
	if image is None:
		raise ValueError(f"Could not read image from path: {image_path}")

	ratio = image.shape[0] / 500.0
	original = image.copy()
	resized = imutils.resize(image, height=500)
	return original, resized, ratio


def preprocess_for_edges(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Blur to reduce noise before edge detection.
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 75, 150)
	return edged


def find_document_contour(edged):
	contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contours)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

	for contour in contours:
		perimeter = cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
		if len(approx) == 4:
			return approx

	return None


def perspective_scan(original_image, contour, ratio):
	warped = four_point_transform(original_image, contour.reshape(4, 2) * ratio)
	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	threshold = threshold_local(warped, 11, offset=10, method="gaussian")
	warped = (warped > threshold).astype("uint8") * 255
	return warped


def scan_document_image(image):
	if image is None:
		raise ValueError("Input image is empty")

	ratio = image.shape[0] / 500.0
	original = image.copy()
	resized = imutils.resize(image, height=500)
	edged = preprocess_for_edges(resized)
	document_contour = find_document_contour(edged)

	if document_contour is None:
		raise ValueError("Could not find a 4-point document contour in the image")

	scanned = perspective_scan(original, document_contour, ratio)
	return original, scanned, document_contour, resized


def scan_document_path(image_path):
	original, resized, ratio = load_and_resize_image(image_path)
	edged = preprocess_for_edges(resized)
	document_contour = find_document_contour(edged)

	if document_contour is None:
		raise ValueError("Could not find a 4-point document contour in the image")

	scanned = perspective_scan(original, document_contour, ratio)
	return original, scanned, document_contour, resized


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
	original, scanned, document_contour, resized = scan_document_path(args["image"])
	show_outline(resized, document_contour)
	show_result(original, scanned)


if __name__ == "__main__":
	main()