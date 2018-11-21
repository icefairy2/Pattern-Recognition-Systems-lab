// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <random>


void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

/************LAB 1************************/
void leastMeanSquares() {
	printf("Choose model nr then file:\n");
	int nr_model;
	scanf("%d", &nr_model);
	if (nr_model != 1 && nr_model != 2) {
		printf("Wrong model number!\n");
		return;
	}

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		FILE *f;
		f = fopen(fname, "r");
		if (f == NULL)
		{
			printf("Can't open file for reading.\n");
			break;
		}

		int n;
		fscanf(f, "%d", &n);
		Point* points;
		points = new Point[n];
		float x, y;
		float minx = FLT_MAX, miny = FLT_MAX;
		float maxx = 0, maxy = 0;
		for (int i = 0; i < n; i++) {
			fscanf(f, "%f%f", &x, &y);
			if (x < minx) {
				minx = x;
			}
			if (y < miny) {
				miny = y;
			}
			if (x > maxx) {
				maxx = x;
			}
			if (y > maxy) {
				maxy = y;
			}
			points[i] = Point(x, y);
		}
		fclose(f);

		int width = maxx - minx + 10;
		int height = maxy - miny + 10;
		Mat img(height, width, CV_8UC3, CV_RGB(255, 255, 255));

		float sumx = 0, sumy = 0, sumxy = 0;
		float sumx2 = 0, sumy2minx2 = 0;

		for (int i = 0; i < n; i++) {
			points[i].x -= minx;
			points[i].y -= miny;
			img.at<Vec3b>(int(points[i].y), int(points[i].x)) = Vec3b(0, 0, 0);
			sumx += points[i].x;
			sumy += points[i].y;
			sumxy += points[i].x * points[i].y;
			sumx2 += points[i].x * points[i].x;
			sumy2minx2 += points[i].y * points[i].y - points[i].x * points[i].x;
		}

		switch (nr_model) {
		case 1:
			//MODEL 1
			float teta0, teta1;
			teta1 = (float)(n * sumxy - sumx * sumy) / (n*sumx2 - sumx * sumx);
			teta0 = (float)(sumy - teta1 * sumx) / (float)n;
			printf("%f %f\n", teta0, teta1);
			line(img, Point(0, teta0), Point(width, teta0 + teta1 * width), Scalar(0, 0, 255));
			break;
		case 2:
			//MODEL 2
			float beta, ro;
			beta = -0.5 * atan2(2 * sumxy - 2.0 / (float)n * sumx * sumy, sumy2minx2 + 1.0 / (float)n * sumx*sumx - 1.0 / (float)n * sumy*sumy);
			ro = 1.0 / (float)n * (cos(beta) * sumx + sin(beta) * sumy);
			printf("%f %f\n", beta, ro);
			line(img, Point(0, ro / sin(beta)), Point(width, (ro - width * cos(beta)) / sin(beta)), Scalar(0, 0, 255));
			break;
		default:
			break;
		}

		imshow("Points", img);
		waitKey();
	}
}

/******************LAB2******************/
#define DISTANCE_TRESHOLD 10
#define CONSTANT_P 0.99
#define PROBABILITY_Q 0.6
#define PROBABILTY_Q_1 0.3

void randomSampleConsensus() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		Point* points = NULL;
		points = new Point[img.rows*img.cols];

		int n = 0;   //Nr of black points
		int i, j;
		for (i = 0; i < img.rows; i++) {
			for (j = 0; j < img.cols; j++) {
				if (img.at<uchar>(i, j) == 0) {
					points[n++] = Point(j, i);
				}
			}
		}

		float N;  //number of required trials
		float T;  //assumed proportion of outliers

		float q = (strstr(fname, "points1")) ? PROBABILTY_Q_1 : PROBABILITY_Q;
		N = log(1.0 - CONSTANT_P) / log(1.0 - pow(q, 2));
		T = q * n;

		std::vector<int> bestModel;
		int selPointInd[2];
		int k = 0;
		while (k < N) {
			//STEP 1
			i = 0;
			int randomNumber;
			bool isDuplicate = false;
			do {
				randomNumber = rand() % n;
				isDuplicate = false;

				if (selPointInd[0] == randomNumber) {
					isDuplicate = true;
					break;
				}

				if (!isDuplicate) {
					selPointInd[i++] = randomNumber;
				}
			} while (i < 2);

			//Define the model S=2
			int a = points[selPointInd[0]].y - points[selPointInd[1]].y;
			int b = points[selPointInd[1]].x - points[selPointInd[0]].x;
			int c = points[selPointInd[0]].x * points[selPointInd[1]].y
				- points[selPointInd[1]].x * points[selPointInd[0]].y;

			//STEP 2
			std::vector<int> model;     //stores the indices from points that are contained in the model
			float dist;
			for (i = 0; i < n; i++) {
				dist = abs(a*points[i].x + b * points[i].y + c) / sqrt(a*a + b * b);
				if (dist <= DISTANCE_TRESHOLD) {
					model.push_back(i);
				}
			}

			//STEP 3
			if (model.size() > bestModel.size()) {
				bestModel = model;
				model.empty();
			}

			//STEP 4
			if (bestModel.size() >= T) {
				break;
			}

			k++;
		}

		line(img, points[selPointInd[0]], points[selPointInd[1]], Scalar(0, 0, 255));

		imshow("RANSAC", img);
		waitKey();
	}
}

/******************LAB3******************/
#define WINDOW_SIZE 3   //=> the window is (3+1+3) => 7x7

struct peak {
	int theta, ro, hval;
	bool operator < (const peak& o) const {
		return hval > o.hval;
	}
};

void houghTransform()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat imgColor = imread(fname, CV_LOAD_IMAGE_COLOR);

		int D = (int)sqrt((double)(img.rows*img.rows + img.cols*img.cols));

		Mat Hough(360, D + 1, CV_32SC1, Scalar(0)); //matrix with int values

		int i, j;

		//hough accumulator
		int ro, theta;
		double rad;
		for (i = 0; i < img.rows; i++) {
			for (j = 0; j < img.cols; j++) {
				if (img.at<uchar>(i, j) == 255) {
					for (int theta = 0; theta < 360; theta++) {
						rad = (double)theta * PI / 180.0;
						ro = j * cos(rad) + i * sin(rad);
						if (ro >= 0 && ro < D) {
							Hough.at<int>(theta, ro)++;
						}
					}
				}
			}
		}

		//finding the maximum for normalization
		int maxHough = 0;
		for (i = 0; i < Hough.rows; i++) {
			for (j = 0; j < Hough.cols; j++) {
				if (Hough.at<int>(i, j) > maxHough) {
					maxHough = Hough.at<int>(i, j);
				}
			}
		}

		Mat houghImg;
		Hough.convertTo(houghImg, CV_8UC1, 255.f / maxHough);

		//we have the image here
		//imshow("Hough", houghImg);
		//waitKey();

		//filtering out the lines
		int window_i, window_j;
		bool local_maxima;
		std::vector<peak> maximas;

		//go over the accumulator
		for (i = 0; i < Hough.rows; i++) {
			for (j = WINDOW_SIZE; j < Hough.cols - WINDOW_SIZE; j++) {
				local_maxima = true;
				//go over the window
				for (window_i = i - WINDOW_SIZE; window_i <= i + WINDOW_SIZE; window_i++) {
					//i is 
					if (window_i < 0) {
						window_i += Hough.rows;
					}
					if (window_i >= Hough.rows) {
						window_i -= Hough.rows;
					}
					for (window_j = j - WINDOW_SIZE; window_j <= j + WINDOW_SIZE; window_j++) {
						//                if (window_j >= 0 && window_j < Hough.cols) {
						if (Hough.at<int>(i, j) < Hough.at<int>(window_i, window_j)) {
							local_maxima = false;
							break;

						}
						//  }
					}
					if (!local_maxima) {
						break;
					}
				}
				if (local_maxima) {
					maximas.push_back(peak{ i, j, Hough.at<int>(i, j) });
				}
			}
		}

		std::sort(maximas.begin(), maximas.end());
		for (i = 0; i < 10; i++) {
			rad = (double)maximas[i].theta * PI / 180.0;
			line(imgColor, Point(0, maximas[i].ro / sin(rad)),
				Point(img.cols, (maximas[i].ro - img.cols * cos(rad)) / sin(rad)),
				Scalar(0, 0, 255));
		}


		imshow("Hough", houghImg);
		imshow("Image", imgColor);
		waitKey();
	}
}

/******************LAB4******************/
Mat distanceTransform(Mat img) {
	Mat distanceTr;
	img.copyTo(distanceTr);

	int mask_k_upper[] = { -1, -1, -1, 0 };
	int mask_l_upper[] = { -1, 0, 1, -1 };
	int mask_upper_weights[] = { 3, 2, 3, 2 };

	int mask_k_lower[] = { 0, 1, 1, 1 };
	int mask_l_lower[] = { 1, -1, 0, 1 };
	int mask_lower_weights[] = { 2, 3, 2, 3 };

	int i, j, k;
	int distance;

	//LEFT->RIGHT, TOP->BOTTOM
	for (i = 1; i < distanceTr.rows - 1; i++) {
		for (j = 1; j < distanceTr.cols - 1; j++) {
			for (k = 0; k < 4; k++) {
				distance = distanceTr.at<uchar>(i + mask_k_upper[k], j + mask_l_upper[k]) + mask_upper_weights[k];
				if (distance < distanceTr.at<uchar>(i, j)) {
					distanceTr.at<uchar>(i, j) = distance;
				}
			}
		}
	}

	//RIGHT->LEFT, BOTTOM->TOP
	for (i = distanceTr.rows - 2; i > 0; i--) {
		for (j = distanceTr.cols - 2; j > 0; j--) {
			for (k = 0; k < 4; k++) {
				distance = distanceTr.at<uchar>(i + mask_k_lower[k], j + mask_l_lower[k]);
				distance += mask_lower_weights[k];
				if (distance < distanceTr.at<uchar>(i, j)) {
					distanceTr.at<uchar>(i, j) = distance;
				}
			}
		}
	}

	return distanceTr;
}

void showDistanceTransform() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat distanceTr = distanceTransform(img);

		imshow("Image", img);
		imshow("DT", distanceTr);
		waitKey();
	}
}

void patternMatching() {
	char fnameTempl[] = "Files/images_DT_PM/PatternMatching/template.bmp";
	Mat templ = imread(fnameTempl, CV_LOAD_IMAGE_GRAYSCALE);

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		Mat distanceTr = distanceTransform(img);

		imshow("DT", distanceTr);
		waitKey();

		int sum = 0;
		int n = 0;

		int i, j;
		for (i = 0; i < templ.rows; i++) {
			for (j = 0; j < templ.cols; j++) {
				if (templ.at<uchar>(i, j) == 0) {
					sum += distanceTr.at<uchar>(i, j);
					n++;
				}
			}
		}

		float avg = (float)sum / n;
		printf("Score: %f\n", avg);
	}
}

/******************LAB5******************/
#define P_FACES 400
#define IMG_ROWS 19
#define IMG_COLS 19
#define N_FEATURES IMG_ROWS*IMG_COLS

void dataAnalysis() {
	Mat features = Mat::zeros(P_FACES, N_FEATURES, CV_8UC1);
	//Read 400 images
	char folder[256] = "Files/images_faces";
	char fname[256];

	int i, j;
	int p;
	Mat img;
	for (int p = 1; p <= P_FACES; p++) {
		sprintf(fname, "%s/face%05d.bmp", folder, p);
		img = imread(fname, 0);
		//compute feature matrix
		for (i = 0; i < IMG_ROWS; i++) {
			for (j = 0; j < IMG_COLS; j++) {
				features.at<uchar>(p - 1, IMG_COLS*i + j) = img.at<uchar>(i, j);
			}
		}
	}

	//compute means
	double means[N_FEATURES];
	for (i = 0; i < N_FEATURES; i++) {
		means[i] = 0;
	}
	for (i = 0; i < N_FEATURES; i++) {
		for (p = 0; p < P_FACES; p++) {
			means[i] += features.at<uchar>(p, i);
		}
		means[i] /= P_FACES;
	}

	std::ofstream meansFile("Files/images_faces/means.csv");

	for (i = 0; i < N_FEATURES; i++) {
		meansFile << means[i] << ',';
	}

	//compute standard deviations
	double stddev[N_FEATURES];
	for (i = 0; i < N_FEATURES; i++) {
		stddev[i] = 0;
	}
	for (i = 0; i < N_FEATURES; i++) {
		for (p = 0; p < P_FACES; p++) {
			stddev[i] += pow(features.at<uchar>(p, i) - means[i], 2);
		}
		stddev[i] /= P_FACES;
		stddev[i] = sqrt(stddev[i]);
	}

	//compute covariance matrix
	Mat covariance = Mat::zeros(N_FEATURES, N_FEATURES, CV_32FC1);
	for (i = 0; i < N_FEATURES; i++) {
		for (j = i; j < N_FEATURES; j++) {
			for (p = 0; p < P_FACES; p++) {
				covariance.at<float>(i, j) += (features.at<uchar>(p, i) - means[i]) * (features.at<uchar>(p, j) - means[j]);
			}
			covariance.at<float>(i, j) /= (float)P_FACES;
			covariance.at<float>(j, i) = covariance.at<float>(i, j);
		}
	}

	std::ofstream covFile("Files/images_faces/covariance.csv");

	for (i = 0; i < N_FEATURES; i++) {
		for (j = 0; j < N_FEATURES; j++) {
			covFile << covariance.at<float>(i, j) << ',';
		}
		covFile << '\n';
	}

	//compute correlation coeffiecients
	Mat correlation = Mat::zeros(N_FEATURES, N_FEATURES, CV_32FC1);
	for (i = 0; i < N_FEATURES; i++) {
		for (j = i; j < N_FEATURES; j++) {
			correlation.at<float>(i, j) = covariance.at<float>(i, j) / (stddev[i] * stddev[j]);
			correlation.at<float>(j, i) = correlation.at<float>(i, j);
		}
	}

	std::ofstream corrFile("Files/images_faces/correlation.csv");

	for (i = 0; i < N_FEATURES; i++) {
		for (j = 0; j < N_FEATURES; j++) {
			corrFile << correlation.at<float>(i, j) << ',';
		}
		corrFile << '\n';
	}

	//display correlation chart
	//left eye: row 5 col 4
	int left_eye = 5 * 19 + 4;
	//right eye: row 5 col 14
	int right_eye = 5 * 19 + 14;
	Mat chart = Mat(256, 256, CV_8UC1, Scalar(255));
	for (p = 0; p < P_FACES; p++) {
		chart.at<uchar>(features.at<uchar>(p, left_eye), features.at<uchar>(p, right_eye)) = 0;
	}
	printf("Covariance: %f\n", covariance.at<float>(left_eye, right_eye));
	printf("Correlation: %f\n", correlation.at<float>(left_eye, right_eye));
	imshow("Chart", chart);
	waitKey();
}

/******************LAB6******************/
//Struct for storing patterns for k-means clustering
struct Pattern {
	std::vector<int> features;
	int cluster;
};

//Calculates the means of k clusters
//Returns: a vector of means, dimension K
//Param K: number of clusters
//Param n: number of patterns
//Param d: number of features
//Param patterns: set of patterns of type Pattern struct, their corresponding cluster 
//					will be calculated and stored in the struct
std::vector<Pattern> kmeans(int K, std::vector<Pattern> &patterns) {
	std::vector<Pattern> means;

	int i, j, dn;
	int n = patterns.size();
	int d = patterns.at(0).features.size();

	std::default_random_engine generator;
	generator.seed(time(NULL));
	std::uniform_int_distribution<int> distribution(0, n - 1);

	//1. INITIALIZE
	int randint;
	for (i = 0; i < K; i++) {
		randint = distribution(generator);
		std::vector<int> features;
		for (dn = 0; dn < d; dn++) {
			features.push_back(patterns.at(randint).features[dn]);
		}
		Pattern new_mean = { features, i };
		means.push_back(new_mean);
	}

	boolean nochanges = false;
	double distance;
	std::vector<int> nr_patterns;
	for (i = 0; i < K; i++) {
		nr_patterns.push_back(0);
	}

	double minDistance;
	int minClusterID;

	while (!nochanges) {
		nochanges = true;

		//2. ASSIGNMENT
		for (i = 0; i < n; i++) {
			minDistance = DBL_MAX;
			minClusterID = patterns.at(i).cluster;

			for (j = 0; j < K; j++) {
				distance = 0;
				for (dn = 0; dn < d; dn++) {
					distance += pow(patterns.at(i).features[dn] - means.at(j).features[dn], 2);
				}
				distance = sqrt(distance);

				if (minDistance > distance) {
					minDistance = distance;
					minClusterID = j;
				}
			}

			if (minClusterID != patterns.at(i).cluster)
			{
				patterns.at(i).cluster = minClusterID;
				nochanges = false;
			}
		}

		//3. UPDATE MEANS
		for (i = 0; i < K; i++) {
			nr_patterns.at(i) = 0;
			for (dn = 0; dn < d; dn++) {
				means.at(i).features[dn] = 0;
			}
		}

		for (i = 0; i < n; i++) {
			for (dn = 0; dn < d; dn++) {
				means.at(patterns.at(i).cluster).features[dn] += patterns.at(i).features[dn];
			}
			nr_patterns[patterns.at(i).cluster]++;
		}

		for (i = 0; i < K; i++) {
			for (dn = 0; dn < d; dn++) {
				if (nr_patterns.at(i) > 0) {
					means.at(i).features[dn] /= nr_patterns.at(i);
				}
			}
		}
	}

	return means;
}

void voronoi_diag(std::vector<Pattern> &means, std::vector<Pattern> &patterns) {
	double minDistance;
	int minClusterID;
	double distance;

	int K = means.size();
	int n = patterns.size();
	int d = patterns.at(0).features.size();

	int i, j, dn;
	for (i = 0; i < n; i++) {
		minDistance = DBL_MAX;
		minClusterID = patterns.at(i).cluster;

		for (j = 0; j < K; j++) {
			distance = 0;
			for (dn = 0; dn < d; dn++) {
				distance += pow(patterns.at(i).features[dn] - means.at(j).features[dn], 2);
			}
			distance = sqrt(distance);

			if (minDistance > distance) {
				minDistance = distance;
				minClusterID = j;
			}
		}

		if (minClusterID != patterns.at(i).cluster)
		{
			patterns.at(i).cluster = minClusterID;
		}
	}
}

void kmeansPoints() {
	printf("Please choose one of the points*.bmp pictures\n");
	//nr of clusters
	const int K = 3;
	printf("Number of clusters is: %d\n", K);

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		std::vector<Pattern> patterns;
		std::vector<Pattern> patterns_all;
		std::vector<Point2d> points;

		int i, j;
		for (i = 0; i < img.rows; i++) {
			for (j = 0; j < img.cols; j++) {

				//there are two features for the points: x position and y position
				std::vector<int> features;
				features.push_back(j);
				features.push_back(i);

				Pattern pattern = { features, 0 };

				if (img.at<uchar>(i, j) == 0) {
					Point2d point(j, i);
					points.push_back(point);

					patterns.push_back(pattern);
				}

				patterns_all.push_back(pattern);
			}
		}

		std::vector<Pattern> means = kmeans(K, patterns);

		Mat img_color(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));

		//calculate K random colors
		Vec3b colors[K];
		std::default_random_engine gen;
		gen.seed(time(NULL));
		std::uniform_int_distribution<int> distribution(0, 255);

		for (int i = 0; i < K; i++) {
			colors[i] = { (uchar)distribution(gen), (uchar)distribution(gen), (uchar)distribution(gen) };
		}

		int n = patterns.size();

		//color the points in the clusters the same for the same cluster
		for (i = 0; i < n; i++) {
			img_color.at<Vec3b>(points[i].y, points[i].x) = colors[patterns.at(i).cluster];
		}

		//create the voronoi diagram
		Mat voronoi(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));

		voronoi_diag(means, patterns_all);

		for (i = 0; i < img.rows; i++) {
			for (j = 0; j < img.cols; j++) {
				n = i * img.cols + j;
				voronoi.at<Vec3b>(i, j) = colors[patterns_all.at(n).cluster];
			}
		}

		imshow("Orig", img);
		imshow("Color", img_color);
		imshow("Voronoi", voronoi);
		waitKey();
	}
}

void kmeansGrayScale() {
	//nr of clusters
	const int K = 6;
	printf("Number of clusters is: %d\n", K);

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		std::vector<Pattern> patterns;

		int i, j;
		for (i = 0; i < img.rows; i++) {
			for (j = 0; j < img.cols; j++) {

				//there is one feature for a grayscale image: the intensity
				std::vector<int> features;
				features.push_back(img.at<uchar>(i, j));

				Pattern pattern = { features, 0 };

				patterns.push_back(pattern);
			}
		}

		std::vector<Pattern> means = kmeans(K, patterns);

		Mat img_res(img.rows, img.cols, CV_8UC1);

		voronoi_diag(means, patterns);

		int n;
		for (i = 0; i < img.rows; i++) {
			for (j = 0; j < img.cols; j++) {
				n = i * img.cols + j;
				img_res.at<uchar>(i, j) = means.at(patterns.at(n).cluster).features[0];
			}
		}

		imshow("Orig", img);
		imshow("Res", img_res);
		waitKey();
	}
}

void kmeansColor() {
	//nr of clusters
	const int K = 3;
	printf("Number of clusters is: %d\n", K);

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);

		std::vector<Pattern> patterns;

		int i, j;
		for (i = 0; i < img.rows; i++) {
			for (j = 0; j < img.cols; j++) {

				//there is one feature for a grayscale image: the intensity
				std::vector<int> features;
				features.push_back(img.at<Vec3b>(i, j)[0]);
				features.push_back(img.at<Vec3b>(i, j)[1]);
				features.push_back(img.at<Vec3b>(i, j)[2]);

				Pattern pattern = { features, 0 };

				patterns.push_back(pattern);
			}
		}

		std::vector<Pattern> means = kmeans(K, patterns);

		Mat img_res(img.rows, img.cols, CV_8UC3);

		voronoi_diag(means, patterns);

		int n;
		for (i = 0; i < img.rows; i++) {
			for (j = 0; j < img.cols; j++) {
				n = i * img.cols + j;
				img_res.at<Vec3b>(i, j)[0] = means.at(patterns.at(n).cluster).features[0];
				img_res.at<Vec3b>(i, j)[1] = means.at(patterns.at(n).cluster).features[1];
				img_res.at<Vec3b>(i, j)[2] = means.at(patterns.at(n).cluster).features[2];
			}
		}

		imshow("Orig", img);
		imshow("Res", img_res);
		waitKey();
	}
}

/******************LAB7******************/
void principalCompAn2() {
	//1.
	FILE *f;
	f = fopen("Files/data_PCA/pca2d.txt", "r");
	if (f == NULL)
	{
		printf("Can't open file for reading.\n");
	}

	int n;
	int d;
	fscanf(f, "%d%d", &n, &d);

	Mat F(n, d, CV_64FC1);

	int i, j;
	double number;
	for (i = 0; i < n; i++) {
		for (j = 0; j < d; j++) {
			fscanf(f, "%lf", &F.at<double>(i, j));
		}
		fscanf(f, "\n");
	}
	fclose(f);

	//2.
	Mat X(n, d, CV_64FC1);
	double* means = new double[d];

	for (i = 0; i < d; i++) {
		means[i] = 0;
		for (j = 0; j < n; j++) {
			means[i] += F.at<double>(j, i);
		}
		means[i] /= (double)n;
	}

	for (i = 0; i < n; i++) {
		for (j = 0; j < d; j++) {
			X.at<double>(i, j) = F.at<double>(i, j) - means[j];
		}
	}

	//3.
	Mat C = X.t()*X / (n - 1);

	//4.
	Mat Lambda, Q;
	eigen(C, Lambda, Q);
	Q = Q.t();

	//5.
	for (i = 0; i < d; i++) {
		printf("Eigenvalue %d: %lf\n", i, Lambda.at<double>(i));
	}

	//6.
	int k = 2;
	Mat Qk(d, k, CV_64FC1);

	for (i = 0; i < d; i++) {
		for (j = 0; j < k; j++) {
			Qk.at<double>(i, j) = Q.at<double>(i, j);
		}
	}

	//coefficients
	Mat Xpca = X * Qk;
	//approximate
	Mat Xktld = Xpca * Qk.t();

	//7.
	double MAD = 0;
	for (i = 0; i < n; i++) {
		for (j = 0; j < d; j++) {
			MAD += abs(X.at<double>(i, j) - Xktld.at<double>(i, j));
		}
	}
	MAD /= n * d;
	printf("MAD: %lf\n", MAD);

	//8.
	double* min = new double[k];
	double* max = new double[k];
	for (i = 0; i < k; i++) {
		min[i] = DBL_MAX;
		max[i] = DBL_MIN;
		for (j = 0; j < n; j++) {
			if (min[i] > Xpca.at<double>(j, i)) {
				min[i] = Xpca.at<double>(j, i);
			}
			if (max[i] < Xpca.at<double>(j, i)) {
				max[i] = Xpca.at<double>(j, i);
			}
		}
	}
	for (i = 0; i < k; i++) {
		printf("Min for column %d: %lf\n", i, min[i]);
		printf("Max for column %d: %lf\n", i, max[i]);
	}

	//9.
	int height = (int)(max[0] - min[0] + 1);
	int width = (int)(max[1] - min[1] + 1);

	Mat normie(n, k, CV_32SC1);
	for (i = 0; i < Xpca.rows; i++) {
		for (j = 0; j < Xpca.cols; j++) {
			normie.at<int>(i, j) = (int)(Xpca.at<double>(i, j) - min[j]);
		}
	}

	Mat donut(height, width, CV_8UC1, Scalar(255));
	for (i = 0; i < n; i++) {
		donut.at<uchar>(normie.at<int>(i, 0), normie.at<int>(i, 1)) = 0;
	}

	imshow("Donut", donut);
	waitKey();
}

void principalCompAn3() {
	//1.
	FILE *f;
	f = fopen("Files/data_PCA/pca3d.txt", "r");
	if (f == NULL)
	{
		printf("Can't open file for reading.\n");
	}

	int n;
	int d;
	fscanf(f, "%d%d", &n, &d);

	Mat F(n, d, CV_64FC1);

	int i, j;
	double number;
	for (i = 0; i < n; i++) {
		for (j = 0; j < d; j++) {
			fscanf(f, "%lf", &F.at<double>(i, j));
		}
		fscanf(f, "\n");
	}
	fclose(f);

	//2.
	Mat X(n, d, CV_64FC1);
	double* means = new double[d];

	for (i = 0; i < d; i++) {
		means[i] = 0;
		for (j = 0; j < n; j++) {
			means[i] += F.at<double>(j, i);
		}
		means[i] /= (double)n;
	}

	for (i = 0; i < n; i++) {
		for (j = 0; j < d; j++) {
			X.at<double>(i, j) = F.at<double>(i, j) - means[j];
		}
	}

	//3.
	Mat C = X.t()*X / (n - 1);

	//4.
	Mat Lambda, Q;
	eigen(C, Lambda, Q);
	Q = Q.t();

	//5.
	for (i = 0; i < d; i++) {
		printf("Eigenvalue %d: %lf\n", i, Lambda.at<double>(i));
	}

	//6.
	int k = 3;
	Mat Qk(d, k, CV_64FC1);

	for (i = 0; i < d; i++) {
		for (j = 0; j < k; j++) {
			Qk.at<double>(i, j) = Q.at<double>(i, j);
		}
	}

	//coefficients
	Mat Xpca = X * Qk;
	//approximate
	Mat Xktld = Xpca * Qk.t();

	//7.
	double MAD = 0;
	for (i = 0; i < n; i++) {
		for (j = 0; j < d; j++) {
			MAD += abs(X.at<double>(i, j) - Xktld.at<double>(i, j));
		}
	}
	MAD /= n * d;
	printf("MAD: %lf\n", MAD);

	//8.
	double* min = new double[k];
	double* max = new double[k];
	for (i = 0; i < k; i++) {
		min[i] = DBL_MAX;
		max[i] = DBL_MIN;
		for (j = 0; j < n; j++) {
			if (min[i] > Xpca.at<double>(j, i)) {
				min[i] = Xpca.at<double>(j, i);
			}
			if (max[i] < Xpca.at<double>(j, i)) {
				max[i] = Xpca.at<double>(j, i);
			}
		}
	}
	for (i = 0; i < k; i++) {
		printf("Min for column %d: %lf\n", i, min[i]);
		printf("Max for column %d: %lf\n", i, max[i]);
	}

	//9.
	int height = (int)(max[0] - min[0] + 1);
	int width = (int)(max[1] - min[1] + 1);

	Mat normie(n, k, CV_32SC1);
	for (i = 0; i < Xpca.rows; i++) {
		for (j = 0; j < Xpca.cols - 1; j++) {
			normie.at<int>(i, j) = (int)(Xpca.at<double>(i, j) - min[j]);
		}
	}

	for (i = 0; i < n; i++) {
		normie.at<int>(i, 2) = (int)((Xpca.at<double>(i, 2) - min[2]) / (max[2] - min[2]) * 255);
	}

	Mat donut(height, width, CV_8UC1, Scalar(255));
	for (i = 0; i < n; i++) {
		donut.at<uchar>(normie.at<int>(i, 1), normie.at<int>(i, 0)) = (uchar)normie.at<int>(i, 2);
	}

	imshow("Lena", donut);
	waitKey();
}

/******************LAB8******************/

#define NR_OF_BINS 8
#define NR_OF_NEIGHBORS 7
#define NR_OF_IMAGES 672
#define NR_OF_CLASSES 6

#define NR_OF_TEST_IMAGES 85

const int nrDim = NR_OF_BINS * 3;

struct FeatureDist {
	double dist;
	int nr_class;
	bool operator < (const FeatureDist& o) const {
		return dist > o.dist;
	}
};

void calcHist(Mat img, int* hist) {
	int i, j;

	Vec3i color_hist[256];
	for (i = 0; i < 256; i++) {
		color_hist[i] = Vec3i(0, 0, 0);
	}

	for (i = 0; i < img.rows; i++) {
		for (j = 0; j < img.cols; j++) {
			color_hist[img.at<Vec3b>(i, j)[0]][0]++;
			color_hist[img.at<Vec3b>(i, j)[1]][1]++;
			color_hist[img.at<Vec3b>(i, j)[2]][2]++;
		}
	}

	Vec3i compr_color_hist[NR_OF_BINS];
	for (i = 0; i < NR_OF_BINS; i++) {
		compr_color_hist[i] = Vec3i(0, 0, 0);
	}

	float chunk_size = 256.f / NR_OF_BINS;

	for (i = 0; i < NR_OF_BINS; i++) {
		for (j = 0; j < 256; j++) {
			if (j >= (i*chunk_size) && j < ((i + 1)*chunk_size)) {
				compr_color_hist[i] += color_hist[j];
			}
		}
	}

	//might store it as |  B  |  G  |  R  |  
	for (i = 0; i < 3; i++) {
		for (j = 0; j < NR_OF_BINS; j++) {
			hist[i*NR_OF_BINS + j] = compr_color_hist[j][i];
		}
	}
}

void knnClassifier() {
	char classes[NR_OF_CLASSES][10] =
	{ "beach", "city", "desert", "forest", "landscape", "snow" };

	int c, i, fileNr;
	int rowX = 0;
	char fname[1000];

	Mat X(NR_OF_IMAGES, nrDim, CV_32FC1);
	int Y[NR_OF_IMAGES];
	int hist[nrDim];

	for (c = 0; c < NR_OF_CLASSES; c++) {
		fileNr = 0;
		while (1) {
			sprintf(fname, "Files/images_KNN/train/%s/%06d.jpeg", classes[c], fileNr++);
			Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
			if (img.cols == 0) break;

			//calculate the histogram in hist
			calcHist(img, hist);

			for (int d = 0; d < nrDim; d++)
				X.at<float>(rowX, d) = hist[d];
			Y[rowX] = c;
			rowX++;
		}
	}

	//number of training files
	int n = rowX;

	rowX = 0;
	int Y_test[NR_OF_IMAGES];
	int Y_assign_test[NR_OF_IMAGES];
	int max, max_class;
	FeatureDist* distances = (FeatureDist*)malloc(n * sizeof(FeatureDist));
	int class_hist[NR_OF_CLASSES];
	double sum;

	for (c = 0; c < NR_OF_CLASSES; c++) {
		fileNr = 0;
		while (1) {
			sprintf(fname, "Files/images_KNN/test/%s/%06d.jpeg", classes[c], fileNr++);
			Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
			if (img.cols == 0) break;

			//calculate the histogram in hist
			calcHist(img, hist);

			for (i = 0; i < n; i++) {
				//calculate distance
				sum = 0;
				for (int k = 0; k < nrDim; k++) {
					sum += pow(X.at<float>(i, k) - hist[k], 2);
				}
				sum = sqrt(sum);

				distances[i].dist = sum;
				distances[i].nr_class = Y[i];
			}

			FeatureDist aux;
			for (i = 0; i < n; i++) {
				for (int j = i + 1; j < n - 1; j++) {
					if (distances[i] < distances[j]) {
						aux = distances[i];
						distances[i] = distances[j];
						distances[j] = aux;
					}
				}
			}

			for (i = 0; i < NR_OF_CLASSES; i++) {
				class_hist[i] = 0;
			}
			for (i = 0; i < NR_OF_NEIGHBORS; i++) {
				class_hist[distances[i].nr_class]++;
			}

			//printf("IMAGE %d\n", rowX);

			max = 0;
			max_class = 0;
			for (i = 0; i < NR_OF_CLASSES; i++) {
				if (class_hist[i] >= max) {
					max = class_hist[i];
					max_class = i;
				}
			}

			Y_assign_test[rowX] = max_class;
			Y_test[rowX] = c;
			rowX++;
		}
	}
	int n_test = rowX;

	Mat confusion_mat(NR_OF_CLASSES, NR_OF_CLASSES, CV_8UC1, Scalar(0));
	for (i = 0; i < n_test; i++) {
		confusion_mat.at<uchar>(Y_test[i], Y_assign_test[i])++;
	}

	for (i = 0; i < NR_OF_CLASSES; i++) {
		for (int j = 0; j < NR_OF_CLASSES; j++) {
			printf("%d ", confusion_mat.at<uchar>(i, j));
		}
		printf("\n");
	}

	int sum_main_diag = 0;
	int sum_all_elem = 0;
	for (i = 0; i < NR_OF_CLASSES; i++) {
		for (int j = 0; j < NR_OF_CLASSES; j++) {
			sum_all_elem += confusion_mat.at<uchar>(i, j);
			if (i == j) {
				sum_main_diag += confusion_mat.at<uchar>(i, j);
			}
		}
	}

	double ACC = (double)sum_main_diag / (double)sum_all_elem;
	printf("ACC: %lf\n", ACC);
	//has to be 0.5647
	getchar();
	getchar();
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Least Mean Squares\n");
		printf(" 11 - Random Sample Consensus\n");
		printf(" 12 - Hough transform for line detection\n");
		printf(" 13 - Distance transform\n");
		printf(" 14 - Pattern matching\n");
		printf(" 15 - Statistical data analysis\n");
		printf(" 16 - K means clustering points\n");
		printf(" 17 - K means clustering grayscale\n");
		printf(" 18 - K means clustering color\n");
		printf(" 19 - Principal Component Analysis 2d\n");
		printf(" 20 - Principal Component Analysis 3d\n");
		printf(" 21 - K-Nearest Neighbor\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testParcurgereSimplaDiblookStyle(); //diblook style
			break;
		case 4:
			//testColor2Gray();
			testBGR2HSV();
			break;
		case 5:
			testResize();
			break;
		case 6:
			testCanny();
			break;
		case 7:
			testVideoSequence();
			break;
		case 8:
			testSnap();
			break;
		case 9:
			testMouseClick();
			break;
		case 10:
			leastMeanSquares();
			break;
		case 11:
			randomSampleConsensus();
			break;
		case 12:
			houghTransform();
			break;
		case 13:
			showDistanceTransform();
			break;
		case 14:
			patternMatching();
			break;
		case 15:
			dataAnalysis();
			break;
		case 16:
			kmeansPoints();
			break;
		case 17:
			kmeansGrayScale();
			break;
		case 18:
			kmeansColor();
			break;
		case 19:
			principalCompAn2();
			break;
		case 20:
			principalCompAn3();
			break;
		case 21:
			knnClassifier();
			break;
		}
	} while (op != 0);
	return 0;
}