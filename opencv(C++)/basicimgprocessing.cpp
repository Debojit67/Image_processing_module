//g++ -ggdb general.cpp -o general `pkg-config --cflags --libs opencv`
//Reference from opencv.org
/**
 * Topics covered:-

0) Basic Drawing
1)Basic thresholdOperation
2) Smoothening
3) Morphological Operations
4) extract horizontal and vertical lines
5) Zoom in/Zoom out
6) Adding borders to Images
7) Sobel operator
8) Laplacian 
9) Canny Edge Detector
10) Hough Line transform
11) Affine Transform
12) Remapping
13) Histogram Plot
14) Template Matching
15) Contours/Convex Hull/min enclosing circle
16) Image moments
Topics covered:-

*/

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
/////////////////////////////Global declaration///////////////////////////////////////////
int Max_Kernel_Length = 15;
string windowName = "original";
int erosion_kernel_size = 1;
int dilation_kernel_size = 1;
int morph_kernel_size = 1;
int morph_oper_selector = 0;
int opertype = 0;
int orgImg_row = 5266, orgImg_col = 3404;
Mat img(orgImg_row, orgImg_col, CV_16UC4);
Mat gray_img(orgImg_row, orgImg_col, CV_16UC1);
int threstype = 0, thresval = 0;
Mat mat_x(img.size(), CV_32FC1), mat_y(img.size(), CV_32FC1);
int selector = 0,selector1=0;
RNG rng(0xFFFFFFFF);
string arrrr[18]={"0) Basic Drawing","1)Basic thresholdOperation","2) Smoothening","3) Morphological Operations","4) extract horizontal and vertical lines","5) Zoom in/Zoom out","6) Adding borders to Images","7) Sobel operator","8) Laplacian","9) Canny Edge Detector","10) Hough Line transform","11) Affine Transform","12) Remapping","13) Histogram Plot","14) Template Matching","15) Contours","16)min enclosing circle","17) Image moments"};
/////////////////////////////////////////////////////////////////////////////////////////////////////////
static void waterShed();
static void imageMoments();
static void minEnclosing();
static void drawContour();
static void templateMatching();
static void histogramPlot();
static void remapping();
static void Affine();
static void houghLine();
static void cannyDetect();
static void LaplacianOperator();
static void sobelOperator();
static void padding();
static void thresholding();
static void zoom();
static void horzvertLines();
static void MorphOperation();
static void smoothening();
static void basicDrawing();

/////////////////////////////////////API DECLARATION////////////////////////////////////////////////////
static void canny_Contourthres(int, void *);
static void templateMatch(int, void *);
static void updateMap(int &index, Mat &mat_x, Mat &mat_y);
static Mat cannyEdge(Mat gray_img, int cannyratio);
static void cannyEdge(int, void *);
static void zooom(int,void*);
static void thresholdOperation(int, void *);
static void showImage(string windowName, Mat image);
static void morphOperation(int, void *);
static Scalar randomColor(RNG &rng);
static void delay(int timeinSec);
static void Erosion_Test(int, void *);
static void Dilation_Test(int, void *);
//////////////////////////////////////////////////////////////////////////////////////////////////////
static void help()
{
    cout << "########################################################" << endl
         << "When asked for choice, Enter the numbers for following operation " << endl
         << "0) Basic Drawing" << endl
         << "1) Basic thresholdOperation" << endl
         << "2) Smoothening" << endl
         << "3) Morphological Operations" << endl
         << "4) extract horizontal and vertical lines" << endl
         << "5) Zoom in/Zoom out" << endl
         << "6) Adding borders to Images" << endl
         << "7) Sobel operator" << endl
         << "8) Laplacian " << endl
         << "9) Canny Edge Detector" << endl
         << "10) Hough Line transform" << endl
         << "11) Affine Transform" << endl
         << "12) Remapping" << endl
         << "13) Histogram Plot" << endl
         << "14) Template Matching" << endl
         << "15) Contours/Convex Hull/min enclosing circle" << endl
         << "16) Image moments" << endl;
}

int main(int argc, char *argv[])
{
    help();
    if (argc < 1)
    {
        cout << "low param count" << endl;
    }

    if (img.empty())
    {
        cout << "Failed to load";
        cin.get();
        return -1;
    }
   img = imread("./l78cY0u6SM-243.jpg", IMREAD_COLOR);
  //  img = imread("./temp.jpg", IMREAD_COLOR);
    //namedWindow(windowName, CV_WINDOW_AUTOSIZE);
    //showImage(windowName, img);
    if (img.channels() == 3)
        cvtColor(img, gray_img, COLOR_BGR2GRAY);
    else
        gray_img = img;
    int ch;
    cout<<"Enter the operation value of your choice"<<endl;
    cin>>ch;
    cout<<"choice entered: "<<arrrr[ch]<<endl;
    switch(ch){
        case 0:basicDrawing();
        break;
        case 1:thresholding();
        break;
        case 2:smoothening();
        break;
        case 3:MorphOperation();
        break;
        case 4:horzvertLines();
        break;
        case 5:zoom();
        break;
        case 6:padding();
        break;
        case 7:sobelOperator();
        break;
        case 8:LaplacianOperator();
        break;
        case 9:cannyDetect();
        break;
        case 10:houghLine();
        break;
        case 11:Affine();
        break;
        case 12:remapping();
        break;
        case 13:histogramPlot();
        break;
        case 14:templateMatching();
        break;
        case 15:drawContour();
        break;
        case 16:minEnclosing();
        break;
        case 17:imageMoments();
        break;

        default: waterShed();
        break;

    }



    return 0;
}

static void basicDrawing()
{
    int looping = 45, windowSize = 400;windowName="Basic drawing";
    String text2Display = "Half Light";
    Mat dst_img = Mat::zeros(windowSize, windowSize, CV_8UC3);
    Point pt1, pt2;
    int range[8] = {50, 100, 150, 200, 250, 300, 350, 375};
    for (int i = 0; i < looping; i++)
    {
        pt1.x = rng.uniform(range[0], range[4]);
        pt1.y = rng.uniform(range[1], range[2]);
        pt2.x = rng.uniform(range[5], range[6]);
        pt2.y = rng.uniform(range[3], range[7]);

        line(dst_img, pt1, pt2, randomColor(rng), 4, 7);

        ellipse(dst_img, Point(pt1.x, pt1.y), Size(windowSize / 4, windowSize / 16), rng.uniform(0, 360), 0, 360, randomColor(rng), rng.uniform(1, 5), 8, 0);

        putText(dst_img, text2Display, Point(pt2.x, pt2.y), rng.uniform(0, 8), rng.uniform(0, 100) * 0.05 + 0.1, randomColor(rng), rng.uniform(1, 10), 8, false);
        imshow(windowName,dst_img);
    }
}
static void smoothening()
{
    Mat dst_img = Mat::zeros(orgImg_row, orgImg_col, CV_8UC4);
    //blur(cv::InputArray src, cv::OutputArray dst, cv::Size ksize, cv::Point anchor = cv::Point(-1, -1), int borderType = 4)
    for (int i = 1; i <= Max_Kernel_Length; i += 2)
    {
        blur(img, dst_img, Size(i, i), Point(-1, -1), 4);
    }
    windowName = "blur";
    showImage(windowName, dst_img);

    for (int i = 1; i <= Max_Kernel_Length; i += 2)
    {
        medianBlur(img, dst_img, i);
    }
    windowName = "medianblur";

    showImage(windowName, dst_img);
}

static void MorphOperation()
{
    namedWindow("Erosion Test", CV_WINDOW_AUTOSIZE);
    namedWindow("Dilation Test", CV_WINDOW_AUTOSIZE);

    //int createTrackbar(const cv::String &trackbarname, const cv::String &winname, int *value, int count, cv::TrackbarCallback onChange = (cv::TrackbarCallback)0, void *userdata = (void *)0)
    createTrackbar("Enter 0:rect \n 1:cross  \n 2:ellipse \n", "Erosion Test", &opertype, 3, Erosion_Test);
    createTrackbar("Enter Kernel size", "Erosion Test", &erosion_kernel_size, Max_Kernel_Length, Erosion_Test);

    createTrackbar("Enter 0:rect \n 1:cross  \n 2:ellipse \n", "Dilation Test", &opertype, 3, Dilation_Test);
    createTrackbar("Enter Kernel size", "Dilation Test", &dilation_kernel_size, Max_Kernel_Length, Dilation_Test);
    Erosion_Test(0, 0);
    Dilation_Test(0, 0);
    waitKey(0);

    windowName = "Morph Operations";
    namedWindow(windowName, CV_WINDOW_AUTOSIZE);
    createTrackbar("Enter \n 0 for opening \n 1 for closing \n 2 for gradient \n 3 for top hat \n 4 for black hat", windowName, &morph_oper_selector, 3, morphOperation);

    createTrackbar("Enter 0:rect \n 1:cross  \n 2:ellipse \n", windowName, &opertype, 2, morphOperation);
    createTrackbar("Enter Kernel size", windowName, &morph_kernel_size, Max_Kernel_Length, morphOperation);
    morphOperation(0, 0);
    waitKey(0);
}

static void horzvertLines()

{
    showImage(windowName, gray_img);
    Mat bw;
    adaptiveThreshold(gray_img, bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
    showImage(windowName, bw);

    Mat horz_lines = bw.clone();
    Mat vert_lines = bw.clone();

    Mat horz_struct = getStructuringElement(MORPH_CROSS, Size(horz_lines.cols / 40, 1));
    Mat vert_struct = getStructuringElement(MORPH_CROSS, Size(1, vert_lines.cols / 40));

    erode(horz_lines, horz_lines, horz_struct, Point(-1, -1));
    dilate(horz_lines, horz_lines, vert_struct, Point(-1, -1));

    erode(vert_lines, vert_lines, vert_struct, Point(-1, -1));
    dilate(vert_lines, vert_lines, vert_struct, Point(-1, -1));

    windowName = "horizontal Lines";
    showImage(windowName, horz_lines);
    windowName = "vertical LInes";
    showImage(windowName, vert_lines);

    bitwise_not(vert_lines, vert_lines);
    showImage(windowName, vert_lines);

    Mat edge_img;
    windowName = "EDGES";
    adaptiveThreshold(vert_lines, edge_img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -2);
    showImage(windowName, edge_img);
    Mat kernel = Mat::ones(2, 2, CV_8UC1);
    dilate(edge_img, edge_img, kernel);
    showImage(windowName, edge_img);

    Mat smooth_img;
    vert_lines.copyTo(smooth_img);
    blur(smooth_img, smooth_img, Size(2, 2));
    windowName = "smoothened";
    smooth_img.copyTo(vert_lines, edge_img);
    showImage(windowName, vert_lines);
}
static void zoom()
{
    Mat dst_img;
    windowName="zoomin/zoomout";
    
    while(true)
    {  
    cout<<"Enter 0 for zoomin and 1 for zoomout and 2 for breaking out of loop"<<endl;
    cin>>selector1;
    cout<<"enter the % of zoom(1x-4x)"<<endl;
    cin>> selector;
    if(selector1==0)
            {pyrDown(img, dst_img, Size(img.rows / selector, img.cols / selector ));
            showImage(windowName,dst_img);
            }
            else if(selector1==1)
            {
                        pyrUp(img, dst_img, Size(img.rows * selector , img.cols * selector));
                        showImage(windowName,dst_img);
            }
            else
            {
                break;
            }
                      
    }
}
static void thresholding()
{

    windowName = "Thresholding operations";
    namedWindow(windowName, CV_WINDOW_AUTOSIZE);
    createTrackbar("Enter \n 0)Binary Threshold \n 1)Binary Threshold invert \n 2)Truncate \n 3)threshold2zero \n 4)threshold2zeroinvert", windowName, &threstype, 4, thresholdOperation);
    createTrackbar("Enter threshold value", windowName, &thresval, 150, thresholdOperation);
    thresholdOperation(0, 0);
    waitKey(0);
}

static void padding()
{
    int temp = rng.uniform(3, 7);
    Mat kernel = Mat::ones(5, 5, CV_32F);
    Mat dst_img = img;
    int ind = 0;
    for (;;)
    {

        int kernel_size = 3 + 2 * (ind % 5);
        cout << kernel_size << endl;
        kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);
        filter2D(img, dst_img, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
        imshow(windowName, dst_img);

        if (waitKey(500) == 27)
        {
            delay(5);
            break;
        }
        ind++;
    }
    // filter2D(img,dst_img,-1,kernel,Point(-1,-1),0,BORDER_DEFAULT);
    // windowName="bordered";
    // showImage(windowName,dst_img);
    Mat tempimg = Mat::zeros(512, 512, CV_8UC3);
    string text2display = "colored borders";
    putText(tempimg, text2display, Point(128, 100), 6, 2, randomColor(rng), 1, 8, false);

    windowName = "bordered";

    int left = 0.05 * img.rows, top = 0.05 * img.cols, bottom = top, right = left;
    namedWindow(windowName, CV_WINDOW_AUTOSIZE);
    copyMakeBorder(img, dst_img, top, bottom, left, right, BORDER_DEFAULT, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
    showImage(windowName, dst_img);
}
static void sobelOperator()
{
    windowName = "sobel";
    Mat blurredImage;
    int kernelSize = -1, delta = 0, scale = 1;
    cout << "enter scale" << endl;
    cin >> scale;
    cout << "enter delta" << endl;
    cin >> delta;
    cout << "Enter kernel Size" << endl;
    cin >> kernelSize;

    cout << "press k,s,d on keyboard to increase kernel,delta and scale value"<<endl<< " 'r' for reset and   'q'   to show current value of delta,scale and kernel size" << endl;
    while (true)
    {
        GaussianBlur(img, blurredImage, Size(3, 3), 0, 0, BORDER_DEFAULT);
        cvtColor(blurredImage, gray_img, CV_BGR2GRAY);
        Mat grad, gradX, gradY, gradXscaled, gradYscaled;
        Sobel(gray_img, gradX, CV_16S, 1, 0, kernelSize, scale, delta, BORDER_DEFAULT);
        Sobel(gray_img, gradY, CV_16S, 0, 1, kernelSize, scale, delta, BORDER_DEFAULT);

        convertScaleAbs(gradX, gradXscaled);
        convertScaleAbs(gradY, gradYscaled);
        addWeighted(gradX, 0.5, gradY, 0.5, 0, grad, -1);

        imshow(windowName, grad);

        if (waitKey(0) == 27)
        {
            return;
        }
        else if ((char)waitKey(0) == 'k' || (char)waitKey(0) == 'K')
            kernelSize = kernelSize < 30 ? kernelSize + 2 : -1;
        else if ((char)waitKey(00) == 's' || (char)waitKey(0) == 'S')
            scale++;
        else if ((char)waitKey(0) == 'd' || (char)waitKey(0) == 'D')
            delta++;
        //reset
        else if ((char)waitKey(0) == 'R' || (char)waitKey(0) == 'r')
        {
            kernelSize = 1, delta = 0;
            scale = 1;
        }
        else if ((char)waitKey(0) == 'q' || (char)waitKey(0) == 'Q')
        {
            cout << "Scale:" << scale << endl;
            cout << "kernelSize" << kernelSize << endl;
            cout << "delta" << delta << endl;
        }
    }
}
static void LaplacianOperator()
{
    Mat tempImage = img, dst_img, abs_dst;
    windowName = "Laplacian";
    GaussianBlur(img, tempImage, Size(3, 3), 0, 0, BORDER_DEFAULT);
    cvtColor(tempImage, gray_img, CV_BGR2GRAY);

    Laplacian(gray_img, dst_img, CV_16S, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(dst_img, abs_dst);
    showImage(windowName, abs_dst);
}
static void cannyDetect()
{
    Mat dst_img(orgImg_row, orgImg_col, CV_8UC4);
    windowName = "canny edge detection";
    Mat tempImage;
    dst_img = Scalar::all(0);
     const int maxthres = 50;
    cvtColor(img, gray_img, CV_BGR2GRAY);
    /** not using trackbar
     * 
    while(true)
   {
        blur(gray_img,tempImage,Size(3,3),Point(-1,-1));
        Canny(tempImage,dst_img,thresval,maxthres,3);
        imshow(windowName,dst_img);
        if(waitKey(0)==27)
            {return;}
            else if ((char)waitKey(0)=='I' ||(char)waitKey(0)=='i')
            thresval=thresval<maxthres/2 ? thresval++:0;
            else if ((char)waitKey(0)=='D' ||(char)waitKey(0)=='d')
            thresval=thresval>0 ? thresval--:0;
   }*/

    namedWindow(windowName,1);
    createTrackbar("Threshold value:",windowName,&thresval,maxthres,cannyEdge);
    cannyEdge(0,0);
    waitKey(0);
}
static void houghLine()
{

    windowName = "Hough transform";
    cvtColor(img, gray_img, CV_BGR2GRAY);
    Mat edges = cannyEdge(gray_img, 3);
    Mat tempimg;
    cvtColor(edges, tempimg, CV_GRAY2BGR);
    showImage(windowName, tempimg);
    vector<Vec2f> lines;
    HoughLines(edges, lines, 1, CV_PI / 180, 150, 0, 0);

    for (size_t i = 0; i <= lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = rho * cos(theta), b = rho * sin(theta);
        pt1.x = cvRound(a + 1000 * (-sin(theta)));
        pt1.y = cvRound(b + 1000 * (cos(theta)));

        pt2.x = cvRound(a - 1000 * (-sin(theta)));
        pt2.y = cvRound(b - 1000 * (cos(theta)));
        line(tempimg, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }
    showImage(windowName, tempimg);
}
static void Affine()
{
    windowName = "affinetransform";
    Point2f srcpts[3], dstpoints[3];
    srcpts[0] = Point2f(0.f, 0.f);
    srcpts[1] = Point2f(img.cols - 2.f, 0.f);
    srcpts[2] = Point2f(0.f, img.rows - 1.f);
    for (int i = 0; i < 3; i++)
        cout << "srd" << srcpts[i] << endl;
    dstpoints[0] = Point2f(0.f, img.rows * 0.5);
    dstpoints[1] = Point2f(img.cols * 0.85f, img.rows * 0.25f);
    dstpoints[2] = Point2f(img.cols * 0.15f, img.rows * 0.7f);

    Mat warp_Affine;
    warp_Affine = getAffineTransform(srcpts, dstpoints);
    Mat warp_dst = Mat::zeros(img.rows, img.cols, img.type());
    warpAffine(img, warp_dst, warp_Affine, warp_dst.size());
    showImage(windowName, warp_Affine);
    showImage(windowName, warp_dst);
    Mat rotated_warpdst = Mat::zeros(img.rows, img.cols, img.type());
    cout<<"Enter the angle of rotation(-180 to +180 degree)"<<endl;
    int angofrot;
    cin>>angofrot;
    Mat rot_matrix = getRotationMatrix2D(Point(img.rows / 2, img.cols / 2), angofrot, 0.6);
    warpAffine(img, rotated_warpdst, rot_matrix, rotated_warpdst.size());
    windowName = "rotated";
    showImage(windowName, rotated_warpdst);
}
static void remapping()
{
    windowName = "Remapping";
    //Mat mat_x(img.size(),CV_32FC1),mat_y(img.size(),CV_32FC1);
    Mat dst_img;
    int index = 0;
    while (true)
    {
        updateMap(index, mat_x, mat_y);
        remap(img, dst_img, mat_x, mat_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
        imshow(windowName, dst_img);
        if (waitKey(0) == 27)
            break;
    }
}

static void histogramPlot()
{
    Mat redchannel(img.size(), CV_8UC1), bluechannel(img.size(), CV_8UC1), greenchannel(img.size(), CV_8UC1);

    Mat bluehist, redhist, greenhist;

    vector<Mat> bgrplane;
    split(img, bgrplane);
    bluechannel = bgrplane[0];

    int histsize = 256, chan = 1;
    float range[] = {0, 256};
    const float *histRange = {range};
    //calcHist(&bluechannel, chan, Mat(), bluehist, histsize, &histRange, false);
}
static void templateMatching()
{
    windowName = "template matching";
    namedWindow(windowName, 1);
    createTrackbar("Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED", windowName, &selector, 5, templateMatch);
    templateMatch(0, 0);
}
static void drawContour()
{

    windowName = "convex hull/contour";
    img = imread("./test1.png", 1);
    showImage(windowName, img);
    namedWindow(windowName, 1);
    createTrackbar("change the threshold level", windowName, &thresval, 100, canny_Contourthres);
    canny_Contourthres(0, 0);
}
static void minEnclosing()
{

    windowName = "Min enclosing circle/rectangle";
    gray_img = imread("./test1.png", 0);
    Mat dst_img = gray_img.clone();
    blur(gray_img, gray_img, Size(3, 3), Point(-1, -1));
    Canny(gray_img, dst_img, 50, 150);
    //showImage(windowName,dst_img);
    //Mat tempimg=Mat::zeros(Size(gray_img.rows,img.cols),CV_8UC3);
    vector<vector<Point>> contours;
    findContours(dst_img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> contourPoly(contours.size());
    vector<Rect> boundingRectangle(contours.size());
    vector<Point2f> center(contours.size());
    vector<float> radius(contours.size());

    for (size_t i = 0; i < contours.size(); i++)
    {
        approxPolyDP(contours[i], contourPoly[i], 3, true);
        boundingRectangle[i] = boundingRect(contourPoly[i]);
        minEnclosingCircle(contourPoly[i], center[i], radius[i]);
    }
    Mat drawimg = Mat::zeros(dst_img.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size() - 1; i++)
    {
        cout << "center" << center[i] << "            "
             << "radius" << radius[i] << endl;
        // drawContours(drawimg,contourPoly,(int)i,randomColor(rng),2,3);
        // rectangle(drawimg,boundingRectangle[i].tl(),boundingRectangle[i].br(),randomColor(rng),1,2);
        circle(drawimg, center[i], (int)radius[i], randomColor(rng), -1);
    }

    showImage(windowName, drawimg);
}
static void imageMoments()
{
    windowName = "Image Moments";
    gray_img = imread("./moments.jpg", 0);
    Mat dst_img = gray_img.clone();
    Mat drawimg = Mat::zeros(dst_img.size(), CV_8UC3);
    blur(gray_img, gray_img, Size(3, 3), Point(-1, -1));
    Canny(gray_img, dst_img, 50, 150);

    showImage(windowName, dst_img);
    vector<vector<Point2f>> contours;
    findContours(dst_img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<Moments> mu(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
        mu[i] = moments(contours[i]);

    vector<Point2f> mc(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        mc[i] = Point2f(static_cast<float>(mu[i].m00 / (mu[i].m01 + 1e-5)), static_cast<float>(mu[i].m00 / (mu[i].m01 + 1e-5)));
        cout << "points" << static_cast<float>(mu[i].m00 / (mu[i].m01 + 1e-5)) << " , " << static_cast<float>(mu[i].m00 / (mu[i].m01 + 1e-5)) << endl;
    }
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(drawimg, contours, (int)i, randomColor(rng), 2, 2);
        // circle(drawimg,mc[i],4,randomColor(rng),1);
    }
    showImage(windowName, drawimg);
}
static void waterShed()
{
    img = imread("./test1.jpeg", 1);
    windowName = "segmentation";
    Mat dst_img;
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            if (img.at<Vec3b>(i, j) == (Vec3b)(255, 255, 255))
            {
                img.at<Vec3b>(i, j)[0] == 0;
                img.at<Vec3b>(i, j)[1] == 0;
                img.at<Vec3b>(i, j)[2] == 0;
            }
        }
    }
    showImage(windowName, img);
    Mat kernel = (Mat_<float>(3, 3) << 1, 1, 1,
                  1, -8, 1,
                  1, 1, 1);
    filter2D(img, dst_img, CV_32F, kernel, Point(-1, -1));
    showImage(windowName, dst_img);
    Mat sharpened;
    img.convertTo(sharpened, CV_32F);

    Mat res_img = dst_img - sharpened;
    res_img.convertTo(res_img, CV_8UC3);
    showImage(windowName, res_img);

    Mat bin_img;
    cvtColor(res_img, bin_img, CV_BGR2GRAY);
    threshold(bin_img, bin_img, 50, 255, THRESH_BINARY);
    showImage(windowName, bin_img);
}

///**********************************************************************************************************************/////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static Scalar randomColor(RNG &rng)
{
    return Scalar((unsigned)rng, (unsigned(rng) >> 8) & 255, ((unsigned(rng) >> 16) & 255));
}
static void delay(int timeinSecs)
{
    for (int k = 1; k <= timeinSecs; k++)
        for (int i = 0; i < 32767 / 2; i++)
            for (int j = 0; j < 32767; j++)
            {
            }
}

static void Erosion_Test(int, void *)
{
    Mat erosion_dstImage(orgImg_row, orgImg_col, CV_8UC4);
    int erosionType = 0;
    if (opertype == 0)
        erosionType = MORPH_RECT;
    else if (opertype == 1)
        erosionType == MORPH_CROSS;
    else
        erosionType = MORPH_ELLIPSE;
    Mat struct_elem = getStructuringElement(erosionType, Size(2 * erosion_kernel_size + 1, 2 * erosion_kernel_size + 1), Point(-1, -1));
    erode(img, erosion_dstImage, struct_elem);
    showImage("Erosion Test", erosion_dstImage);
    //erode(cv::InputArray src, cv::OutputArray dst, cv::InputArray kernel, cv::Point anchor = cv::Point(-1, -1), int iterations = 1, int borderType = 0, const cv::Scalar &borderValue = morphologyDefaultBorderValue())
}

static void Dilation_Test(int, void *)
{
    Mat dilation_dstImage(orgImg_row, orgImg_col, CV_8UC4);
    int dilationType = 0;
    if (opertype == 0)
        dilationType = MORPH_RECT;
    else if (opertype == 1)
        dilationType == MORPH_CROSS;
    else
        dilationType = MORPH_ELLIPSE;
    Mat struct_elem = getStructuringElement(dilationType, Size(2 * dilation_kernel_size + 1, 2 * dilation_kernel_size + 1), Point(-1, -1));
    dilate(img, dilation_dstImage, struct_elem);
    showImage("Dilation Test", dilation_dstImage);
    //erode(cv::InputArray src, cv::OutputArray dst, cv::InputArray kernel, cv::Point anchor = cv::Point(-1, -1), int iterations = 1, int borderType = 0, const cv::Scalar &borderValue = morphologyDefaultBorderValue())
    waitKey(0);
}
static void morphOperation(int, void *)
{
    Mat morph_dst(orgImg_row, orgImg_col, CV_8UC4);
    Mat struct_elem = getStructuringElement(opertype, Size(2 * morph_kernel_size + 1, 2 * morph_kernel_size + 1), Point(-1, -1));
    morphologyEx(img, morph_dst, morph_oper_selector + 2, struct_elem);
    imshow(windowName, morph_dst);
}
static void thresholdOperation(int, void *)
{
    int maxbinvalue = 255;
    Mat dst_img(orgImg_row, orgImg_col, CV_8UC1);
    threshold(gray_img, dst_img, thresval, maxbinvalue, threstype);
    imshow(windowName, dst_img);
}
static void cannyEdge(int, void *)
{
    Mat dst_img(orgImg_row, orgImg_col, CV_8UC4);
    dst_img = Scalar::all(0);
    Mat tempImage;
    cout<<"threshold value "<<thresval<<endl;
    int thresmax = 100, ratio = 3;
    blur(gray_img, tempImage, Size(3, 3), Point(-1, -1));
    Canny(tempImage, dst_img, thresval, thresval * ratio, 3);
    imshow(windowName,dst_img);
}
static Mat cannyEdge(Mat gray_img, int cannyratio)
{
    Mat dst_img(orgImg_row, orgImg_col, CV_8UC4);
    dst_img = Scalar::all(0);
    Mat tempImage;
    int thresmax = 100;
    blur(gray_img, tempImage, Size(3, 3), Point(-1, -1));
    Canny(tempImage, dst_img, thresval, thresval * cannyratio, 3);
    return dst_img;
}
static void updateMap(int &index, Mat &mat_x, Mat &mat_y)
{
    for (int i = 0; i < mat_x.rows; i++)
    {
        for (int j = 0; j < mat_x.cols; j++)
        {
            switch ((index))
            {
            case 0:
                if (j > mat_x.cols * 0.25 && j < mat_y.cols * 0.75 && i > mat_x.rows * 0.25 && i < mat_y.rows * 0.75)
                {
                    mat_x.at<float>(i, j) = (float)(2 * (j - mat_x.rows * 0.25f) + 0.5f);
                    mat_y.at<float>(i, j) = (float)(2 * (i - mat_y.cols * 0.25f) + 0.5f);
                }
                else
                {
                    mat_x.at<float>(i, j) = 0;
                    mat_y.at<float>(i, j) = 0;
                }
                break;
            case 1:
                mat_x.at<float>(i, j) = (float)j;
                mat_y.at<float>(i, j) = (float)(img.rows - i);
                break;
            case 2:
                mat_x.at<float>(i, j) = (float)(img.cols - j);
                mat_y.at<float>(i, j) = (float)(i);
                break;
            case 3:
                mat_x.at<float>(i, j) = (float)(img.cols - j);
                mat_y.at<float>(i, j) = (float)(img.rows - i);
                break;

            default:
                break;
            }
        }
    }
    index = (index + 1) % 4;
}
static void templateMatch(int, void *)
{
    Mat template_img = imread("template.jpg", IMREAD_COLOR);
    double minval, maxval;
    Point minlocation, maxlocation, matchlocation;
    Mat corrimage(img.rows - template_img.rows + 1, img.cols - template_img.cols + 1, CV_32FC1);
    matchTemplate(img, template_img, corrimage, selector);
    normalize(corrimage, corrimage, 0, 1, NORM_MINMAX, -1, Mat());
    minMaxLoc(corrimage, &minval, &maxval, &minlocation, &maxlocation, Mat());
    //   minMaxLoc(corrimage,&)

    if (selector == 0 || selector == 1)
        matchlocation = minlocation;
    else
        matchlocation = maxlocation;
    Mat tempdispimg = img.clone();
    rectangle(tempdispimg, matchlocation, Point(template_img.cols + matchlocation.x, template_img.rows + matchlocation.y), randomColor(rng), 2, 8, 0);
    rectangle(corrimage, matchlocation, Point(matchlocation.x + template_img.cols, matchlocation.y + template_img.rows), randomColor(rng), 2, 8, 0);
    imshow(windowName, tempdispimg);
    waitKey(0);
    //imshow(windowName,corrimage);
}

static void canny_Contourthres(int, void *)
{

    Mat dst_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);
    blur(gray_img, gray_img, Size(3, 3), Point(-1, -1));

    Canny(gray_img, dst_img, thresval, 2 * thresval);
    vector<vector<Point>> contour;

    findContours(dst_img, contour, RETR_TREE, CHAIN_APPROX_SIMPLE);

    vector<vector<Point>> hull(contour.size());
    for (size_t i = 0; i < contour.size(); i++)
    {
        convexHull(contour[i], hull[i]);
    }
    Mat drawimg = Mat::zeros(dst_img.size(), CV_8UC3);
    cout << "Abc";
    for (size_t i = 0; i < contour.size(); i++)
    {
        drawContours(drawimg, contour, (int)i, randomColor(rng));
        drawContours(drawimg, hull, (int)i, randomColor(rng));
    }
    imshow(windowName, drawimg);
    waitKey(0);
}
static void showImage(string windowName, Mat image)
{
    imshow(windowName, image);
    waitKey(0);
    destroyWindow(windowName);
}