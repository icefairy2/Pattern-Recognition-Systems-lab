// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"


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

        // the fastest approach using the �diblook style�
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
        }
    } while (op != 0);
    return 0;
}