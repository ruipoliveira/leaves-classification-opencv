#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

int main(int argc, char *argv[]){
    Mat src_gray, detected_edges, dst;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 10;
int ratio = 3;
int kernel_size = 3;


    Mat src = imread("../image_leaves/Acer_platanoides.jpg", CV_LOAD_IMAGE_COLOR);

    cvtColor( src, src_gray, CV_BGR2GRAY );

    blur( src_gray, detected_edges, Size(3,3) );


    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

    /// Using Canny's output as a mask, we display our result
    dst = Scalar::all(0);

    src.copyTo( dst, detected_edges);


    if(src.empty())
       return -1;
    namedWindow( "lena", CV_WINDOW_NORMAL );
    imshow("lena", dst);
    waitKey(0);
    return 0;
}



