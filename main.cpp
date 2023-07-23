#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

vector <int> rect1x;
vector <int> rect1y;
vector <int> rect2x;
vector <int> rect2y;
vector<Point> traversePoint;
int total = 9; 
Mat handHist;

Mat rectangles(Mat img) {
	int rows = img.rows, cols = img.cols;

	int a = 3;
	rect1x.clear();
	rect1y.clear();
	rect2x.clear();
	rect2y.clear();
	for (int i = 0; i < 9; i++) {
		if (i % 3 == 0) {
			a += 3;
		}
		rect1x.push_back(a * rows / 20);
	}
	for (int i = 0; i < 9; i++) {
		if (i % 3 == 0) {
			a = 9;
		}
		rect1y.push_back(a++ * cols / 20);
	}
	rect2x = rect1x;
	rect2y = rect1y;

	for_each(rect2x.begin(), rect2x.end(), [](int& n) {n += 10; });
	for_each(rect2y.begin(), rect2y.end(), [](int& m) {m += 10; });

	for (int i = 0; i < total; i++)
		rectangle(img, Point(rect1y[i], rect1x[i]), Point(rect2y[i], rect2x[i]), Scalar(255, 0, 0), 1);
	return img;
}

Mat histogram(cv::Mat img) {
	Mat imgHSV, roi;
	cvtColor(img, imgHSV, COLOR_BGR2HSV);

	Mat imgDest(90, 10, CV_8UC3, cv::Scalar(0, 0, 0));

	for (int i = 0; i < 9; i++) {
		Rect roiRect(rect1y[i], rect1x[i], 10, 10);
		roi = imgHSV(roiRect);
		Rect dstRect(0, i * 10, 10, 10);
		roi.copyTo(imgDest(dstRect));
	}

	int histSize[] = { 180, 256 };
	float range[][2] = { {0, 180}, {0, 256} };
	const float* histRange[] = { range[0], range[1] };
	const int channels[] = { 0, 1 };

	calcHist(&imgDest, 1, channels, Mat(), handHist, 2, histSize, histRange);

	normalize(handHist, handHist, 0, 255, cv::NORM_MINMAX);

	return handHist;
}

Mat maskHist(Mat img, Mat hist) {
	Mat hsv, dest, thresh, btAnd;
	cvtColor(img, hsv, COLOR_BGR2HSV);

	int channels[] = { 0, 1 };
	float ranges[][2] = { {0, 180}, {0, 256} };
	const float* histRange[] = { ranges[0], ranges[1] };

	calcBackProject(&hsv, 1, channels, hist, dest, histRange, 1, true);

	Mat disc = getStructuringElement(MORPH_ELLIPSE, Size(31, 31));
	filter2D(dest, dest, -1, disc);

	double threshVal = 150;
	double maxThresh = 255;
	threshold(dest, thresh, threshVal, maxThresh, THRESH_BINARY);

	vector<Mat> ch = { thresh, thresh, thresh };
	merge(ch, thresh);
	bitwise_and(img, thresh, btAnd);

	return btAnd;
}

vector<vector<Point>> contours(Mat imgMask) {
	Mat grayHistMask, thresh;
	cvtColor(imgMask, grayHistMask, COLOR_BGR2GRAY);
	threshold(grayHistMask, thresh, 0.00, 255.00, 0);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	return contours;
}

Point2f centroid(const vector<Point>& contour) {
	Moments moment = moments(contour);
	if (moment.m00 != 0) {
		Point2f centroid(moment.m10 / moment.m00, moment.m01 / moment.m00);
		return centroid;
	}

	return cv::Point2f(-1, -1);
}

bool isCntEmpty(const Point& point) {
	return (point.x == 0 && point.y == 0);
}

Point farPoint(const vector<Vec4i>& defects, const vector<Point>& contour, const Point2f& centroid) {
	if (!defects.empty() && !isCntEmpty(centroid)) {
		int farthestDefect, cx, cy, maxI;
		double maxi = 0;
		vector <int> res, x, y;
		vector <long long int> xp, yp;
		vector <long double> dist;
		Point farthestPoint;

		for (auto i : defects) {
			res.push_back(i[0]);
		}
		cx = centroid.x;
		cy = centroid.y;
		for (int a = 0; a < res.size(); a++) {
			int i = res[a];
			x.push_back(contour[i].x);
		}
		for (int a = 0; a < res.size(); a++) {
			int i = res[a];
			y.push_back(contour[i].y);
		}

		cout << endl;
		for (int i = 0; i < x.size(); i++) {
			xp.push_back(pow(x[i] - cx, 2));
			yp.push_back(pow(y[i] - cy, 2));
		}
		dist.resize(xp.size());
		for (int i = 0; i < xp.size(); i++) {
			long long int a = xp[i] + yp[i];
			dist[i] = sqrt(a);
			if (dist[i] > maxi) {
				maxI = i;
				maxi = dist[i];
			}
		}
		if (maxI < res.size()) {
			farthestDefect = res[maxI];
			farthestPoint = contour[farthestDefect];
			return farthestPoint;
		}
	}
	return { -1, -1 };
}


void imgOpr(Mat img, Mat handHist) {
	Mat imgMask;
	double maxArea = -1.0;
	Point2f cnt;
	Point far_Point;
	vector <int> hull;
	vector<Point> maxContour;
	vector<Vec4i> defects;
	vector <vector<Point>> allContours;

	imgMask = maskHist(img, handHist);

	erode(imgMask, imgMask, Mat(), Point(-1, -1), 2);
	dilate(imgMask, imgMask, Mat(), Point(-1, -1), 2);
	
	allContours = contours(imgMask);

	for (const auto& contour : allContours) {
		double area = contourArea(contour);
		if (area > maxArea) {
			maxArea = area;
			maxContour = contour;
		}
	}
	cnt = centroid(maxContour);

	circle(img, cnt, 5, Scalar(255, 0, 255), -1);
	if (cnt.x != -1 && cnt.y != -1) {
		convexHull(maxContour, hull, false);
		convexityDefects(maxContour, hull, defects);

		far_Point = farPoint(defects, maxContour, cnt);

		cout << "Centroid : (" << cnt.x << ", " << cnt.y << "), farthest Point : (" << far_Point.x << ", " << far_Point.y << ")";
		int a = far_Point.x, b = far_Point.y;
		SetCursorPos(a, b);
		circle(img, far_Point, 5, Scalar(0, 0, 255), FILLED);

		if (traversePoint.size() < 20)
			traversePoint.push_back(far_Point);

		else {
			traversePoint.erase(traversePoint.begin());
			traversePoint.push_back(far_Point);
		}
	}
}

Mat rescaleImg(Mat img, int perW, int perH) {
	int width = img.cols * perW / 100;
	int height = img.rows * perH / 100;
	Mat newImg;
	resize(img, newImg, Size(width, height), 0, 0, INTER_AREA);
	return newImg;
}

int main() {
	VideoCapture cap(0);
	Mat img;
	bool isHandHistCreated = false;

	while (true) {
		char keyPressed = (char)waitKey(1);
		cap.read(img);
		flip(img, img, 1);
		if ((keyPressed & 0xFF) == static_cast<int>('z')) {
			isHandHistCreated = true;
			handHist = histogram(img);
		}
		else if ((keyPressed & 0xFF) == static_cast<int>('a')) {
			mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
			keyPressed = 'z';
		}
		
		else if ((keyPressed & 0xFF) == static_cast<int>('d')) {
			mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0);
			keyPressed = 'z';
		}
		
		else if ((keyPressed & 0xFF) == static_cast<int>('w')) {
			mouse_event(MOUSEEVENTF_WHEEL, 0, 0, 1 * 120, 0);
			keyPressed = 'z';
		}
		
		else if ((keyPressed & 0xFF) == static_cast<int>('s')) {
			mouse_event(MOUSEEVENTF_WHEEL, 0, 0, -120, 0);
			keyPressed = 'z';
		}
		

		if (isHandHistCreated)
			imgOpr(img, handHist);
		else
			rectangles(img);

		imshow("Webcam", rescaleImg(img, 130, 130));

		if (keyPressed == 27)
			break;
	}
	destroyAllWindows();
	cap.release();
	return 0;
}
