#include <stdio.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/ml/ml.hpp>


using namespace cv;
using namespace std;



/*
* Speeded Up Robust Features
* http://docs.opencv.org/2.4/doc/tutorials/features2d/feature_homography/feature_homography.html
*
*  Ver: http://stackoverflow.com/questions/16724591/opencv-sift-all-of-the-features-of-2-different-insects-are-matching
*
*/
int testSURF(){


  Mat img_object = imread( "../image_leaves/Acer_platanoides.jpg", CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_scene = imread( "../image_leaves_biologia/Acer_platanoides/IMG_0112.JPG", CV_LOAD_IMAGE_GRAYSCALE );

    //imshow( "Good Matches & Object detection", img_object );


  if( !img_object.data || !img_scene.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;

  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_object, keypoints_scene;

  detector.detect( img_object, keypoints_object );
  detector.detect( img_scene, keypoints_scene );

  //-- Step 2: Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;

  Mat descriptors_object, descriptors_scene;

  extractor.compute( img_object, keypoints_object, descriptors_object );
  extractor.compute( img_scene, keypoints_scene, descriptors_scene );

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_object, descriptors_scene, matches );

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_object.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_object.rows; i++ )
  { if( matches[i].distance < 3*min_dist )
     { good_matches.push_back( matches[i]); }
  }

  Mat img_matches;
  drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  //-- Localize the object
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;

  for( int i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }

  Mat H = findHomography( obj, scene, CV_RANSAC );

  //-- Get the corners from the image_1 ( the object to be "detected" )
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
  std::vector<Point2f> scene_corners(4);

  perspectiveTransform( obj_corners, scene_corners, H);

  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
  line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

  //-- Show detected matches
  namedWindow( "Good Matches & Object detection", CV_WINDOW_NORMAL );
  imshow( "Good Matches & Object detection", img_matches );

  waitKey(0);

  return 0;

}

int removeBackground(){
    Mat src_gray, detected_edges, dst;

    int edgeThresh = 1;
    int lowThreshold;
    int const max_lowThreshold = 10;
    int ratio = 3;
    int kernel_size = 3;


    Mat src = imread("../image_leaves_biologia/Acer_platanoides/IMG_0112.JPG", CV_LOAD_IMAGE_COLOR);

    cvtColor( src, src_gray, CV_BGR2GRAY );


    threshold( src_gray, dst, 0,255,CV_THRESH_BINARY|CV_THRESH_OTSU);


    blur( src_gray, src_gray, Size(5,5) );


    //Canny( src_gray, src_gray, lowThreshold, lowThreshold*ratio, kernel_size );

    /// Using Canny's output as a mask, we display our result
    //dst = Scalar::all(0);

    //src.copyTo( dst, detected_edges);


    if(src.empty())
       return -1;
    namedWindow( "lena", CV_WINDOW_NORMAL );
    imshow("lena", dst);
    waitKey(0);
    return 0;
}



int histograma(){
    Mat gray=imread("../image_leaves/Acer_platanoides.jpg",0);
    namedWindow( "Gray", CV_WINDOW_NORMAL );
    imshow( "Gray", gray );

    // Initialize parameters
    int histSize = 256;    // bin size
    float range[] = { 0, 255 };
    const float *ranges[] = { range };

    // Calculate histogram
    MatND hist;
    calcHist( &gray, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false );

    // Show the calculated histogram in command window
    double total;
    total = gray.rows * gray.cols;
    for( int h = 0; h < histSize; h++ )
         {
            float binVal = hist.at<float>(h);
            cout<<" "<<binVal;
         }

    // Plot the histogram
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    for( int i = 1; i < histSize; i++ ) {
    line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
           Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
           Scalar( 255, 0, 0), 2, 8, 0  );
    }

    namedWindow( "Result", 1 );
    imshow( "Result", histImage );

    waitKey(0);
    return 0;

}

/*
Support Vector Machines

http://docs.opencv.org/2.4/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html#introduction-to-support-vector-machines
*/

int svm(){
// Data for visual representation
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);

    // Set up training data
    float labels[4] = {1.0, -1.0, -1.0, -1.0};
    Mat labelsMat(4, 1, CV_32FC1, labels);

    float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
    Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
    CvSVM SVM;
    SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

    Vec3b green(0,255,0), blue (255,0,0);
    // Show the decision regions given by the SVM
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
        {
            Mat sampleMat = (Mat_<float>(1,2) << j,i);
            float response = SVM.predict(sampleMat);

            if (response == 1)
                image.at<Vec3b>(i,j)  = green;
            else if (response == -1)
                 image.at<Vec3b>(i,j)  = blue;
        }

    // Show the training data
    int thickness = -1;
    int lineType = 8;
    circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType);
    circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType);
    circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
    circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType);

    // Show support vectors
    thickness = 2;
    lineType  = 8;
    int c     = SVM.get_support_vector_count();

    for (int i = 0; i < c; ++i)
    {
        const float* v = SVM.get_support_vector(i);
        circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
    }

    imwrite("result.png", image);        // save the image

    imshow("SVM Simple Example", image); // show it to the user
    waitKey(0);

    return 0;
}



void help()
{
	cout << endl << "USAGE: ./car_detect IMAGE.EXTENTION checkcas.xml cas1.xml cas2.xml cas3.xml cas4.xml ..........upto n number of main cascade xml files" << endl;
	cout << endl << "ckeckcas.xml is the one trained with smallest size parameters and the rest are the main cascades" << endl;
}




class cars     //main class
{
	public:	    //variables kept public but precaution taken all over the code

	Mat image_input;          //main input image
	Mat image_main_result;    //the final result
	Mat storage;              //introduced to stop detection of same car more than once

	CascadeClassifier cascade;    //the main cascade classifier
	CascadeClassifier checkcascade;        //a test classifier,car detected by both main and test is stated as car

	int num;

	void getimage(Mat src) //getting the input image
    {

        if(! src.data )
        {
            cout <<  "src not filled" << endl ;
        }

        else
        {
            image_input = src.clone();
			storage = src.clone();              //initialising storage
			image_main_result = src.clone();    //initialising result

            cout << "got image" <<endl;
        }
    }


	void cascade_load(string cascade_string)            //loading the main cascade
	{
		cascade.load(cascade_string);

		if( !cascade.load(cascade_string) )
    	{
        	cout << endl << "Could not load classifier cascade" << endl;

    	}
		else
		{

			cout << "cascade : " << cascade_string << " loaded" << endl;
		}

	}


	void checkcascade_load(string checkcascade_string)               //loading the test/check cascade
	{
		checkcascade.load(checkcascade_string);

		if( !checkcascade.load(checkcascade_string) )
    	{
        	cout << endl << "Could not load classifier checkcascade" << endl;

    	}
		else
		{
			cout<< "checkcascade : " << checkcascade_string << " loaded" << endl;
		}
	}


	void display_input()             // function to display input
	{
		namedWindow("display_input");
		imshow("display_input",image_input);
		waitKey(0);
	}

	void display_output()            //function to display output
	{

		if(!image_main_result.empty() )
        {
			namedWindow("display_output");
			imshow("display_output",image_main_result);
			waitKey(0);
		}
	}

	void setnum()
	{
		num = 0;
	}


	void findcars()                 //main function
	{
    	int i = 0;

		Mat img = storage.clone();
		Mat temp;                    //for region of interest.If a car is detected(after testing) by one classifier,then it will not be available for other one

		if(img.empty() )
        {
			cout << endl << "detect not successful" << endl;
		}
		int cen_x;
		int cen_y;
    	vector<Rect> cars;
    	const static Scalar colors[] =  { CV_RGB(0,0,255),CV_RGB(0,255,0),CV_RGB(255,0,0),CV_RGB(255,255,0),CV_RGB(255,0,255),CV_RGB(0,255,255),CV_RGB(255,255,255),CV_RGB(128,0,0),CV_RGB(0,128,0),CV_RGB(0,0,128),CV_RGB(128,128,128),CV_RGB(0,0,0)};

    	Mat gray;

    	cvtColor( img, gray, CV_BGR2GRAY );

		Mat resize_image(cvRound (img.rows), cvRound(img.cols), CV_8UC1 );

    	resize( gray, resize_image, resize_image.size(), 0, 0, INTER_LINEAR );
    	equalizeHist( resize_image, resize_image );


    	cascade.detectMultiScale( resize_image, cars,1.1,2,0,Size(10,10));                 //detection using main classifier


		for( vector<Rect>::const_iterator main = cars.begin(); main != cars.end(); main++, i++ )
    	{
       		Mat resize_image_reg_of_interest;
        	vector<Rect> nestedcars;
        	Point center;
        	Scalar color = colors[i%8];


			//getting points for bouding a rectangle over the car detected by main
			int x0 = cvRound(main->x);
			int y0 = cvRound(main->y);
			int x1 = cvRound((main->x + main->width-1));
			int y1 = cvRound((main->y + main->height-1));



        	if( checkcascade.empty() )
            	continue;
        	resize_image_reg_of_interest = resize_image(*main);
        	checkcascade.detectMultiScale( resize_image_reg_of_interest, nestedcars,1.1,2,0,Size(30,30));

        	for( vector<Rect>::const_iterator sub = nestedcars.begin(); sub != nestedcars.end(); sub++ )      //testing the detected car by main using checkcascade
        	{
           		center.x = cvRound((main->x + sub->x + sub->width*0.5));        //getting center points for bouding a circle over the car detected by checkcascade
				cen_x = center.x;
			   	center.y = cvRound((main->y + sub->y + sub->height*0.5));
				cen_y = center.y;
				if(cen_x>(x0+15) && cen_x<(x1-15) && cen_y>(y0+15) && cen_y<(y1-15))         //if centre of bounding circle is inside the rectangle boundary over a threshold the the car is certified
				{

					rectangle( image_main_result, cvPoint(x0,y0),
                    	   		cvPoint(x1,y1),
                     	  		color, 3, 8, 0);               //detecting boundary rectangle over the final result



					//masking the detected car to detect second car if present

					Rect region_of_interest = Rect(x0, y0, x1-x0, y1-y0);
					temp = storage(region_of_interest);
					temp = Scalar(255,255,255);

					num = num+1;     //num if number of cars detected

				}
			}

		}


	if(image_main_result.empty() )
    {
		cout << endl << "result storage not successful" << endl;
	}

    }

};





int carDetetion(){


	double t = 0;
    t = (double)cvGetTickCount();              //starting timer
    Mat image1 = imread("/home/roliveira/Documents/1.jpg",1);
	Mat image;
	resize(image1,image,Size(300,150),0,0,INTER_LINEAR);        //resizing image to get best experimental results
	cars detectcars;                      //creating a object


	string checkcas = "/home/roliveira/Documents/test/CAR-DETECTION/checkcas.xml";

	detectcars.getimage(image);           //get the image
	detectcars.setnum();                  //set number of cars detected as 0
	detectcars.checkcascade_load(checkcas);      //load the test cascade

	//Applying various cascades for a finer search.

    string cas1 = "/home/roliveira/Documents/test/CAR-DETECTION/cas1.xml";
    string cas2 = "/home/roliveira/Documents/test/CAR-DETECTION/cas2.xml";
    string cas3 = "/home/roliveira/Documents/test/CAR-DETECTION/cas3.xml";
    string cas4 = "/home/roliveira/Documents/test/CAR-DETECTION/cas4.xml";

    detectcars.cascade_load(cas1);
    detectcars.cascade_load(cas2);
    detectcars.cascade_load(cas3);
    detectcars.cascade_load(cas4);
    detectcars.findcars();



	t = (double)cvGetTickCount() - t;       //stopping the timer

	if(detectcars.num!=0)
	{
		cout << endl << detectcars.num << " cars got detected in = " << t/((double)cvGetTickFrequency()*1000.) << " ms" << endl << endl;
	}
	else
	{
		cout << endl << "cars not found" << endl;
	}

	detectcars.display_output();          //displaying the final result

}


int bagOfFeatruresREAD(){
//to store the input file names
char * filename = new char[100];
//to store the current input image
Mat input;

//To store the keypoints that will be extracted by SIFT
vector<KeyPoint> keypoints;
//To store the SIFT descriptor of current image
Mat descriptor;
//To store all the descriptors that are extracted from all the images.
Mat featuresUnclustered;
//The SIFT feature extractor and descriptor
SiftDescriptorExtractor detector;


sprintf(filename,"/home/roliveira/Documents/leaves-classification/image_leaves/Acer_platanoides.jpg",1);
//open the file
input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale
//detect feature points
detector.detect(input, keypoints);
//compute the descriptors for each keypoint
detector.compute(input, keypoints,descriptor);
//put the all feature descriptors in a single Mat object
featuresUnclustered.push_back(descriptor);
//print the percentage



//Construct BOWKMeansTrainer
//the number of bags
int dictionarySize=200;
//define Term Criteria
TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
//retries number
int retries=1;
//necessary flags
int flags=KMEANS_PP_CENTERS;
//Create the BoW (or BoF) trainer
BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
//cluster the feature vectors
Mat dictionary=bowTrainer.cluster(featuresUnclustered);
//store the vocabulary
FileStorage fs("dictionary.yml", FileStorage::WRITE);
fs << "vocabulary" << dictionary;
fs.release();


}


int testBag(){
   //prepare BOW descriptor extractor from the dictionary
    Mat dictionary;
    FileStorage fs("dictionary.yml", FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    fs.release();

    //create a nearest neighbor matcher
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    //create Sift feature point extracter
    Ptr<FeatureDetector> detector(new SiftFeatureDetector());
    //create Sift descriptor extractor
    Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor bowDE(extractor,matcher);
    //Set the dictionary with the vocabulary we created in the first step
    bowDE.setVocabulary(dictionary);

    //To store the image file name
    char * filename = new char[100];
    //To store the image tag name - only for save the descriptor in a file
    char * imageTag = new char[10];

    //open the file to write the resultant descriptor
    FileStorage fs1("descriptor.yml", FileStorage::WRITE);

    //the image file with the location. change it according to your image file location
    sprintf(filename,"/home/roliveira/Documents/leaves-classification/image_leaves/Acer_platanoides.jpg");
    //read the image
    Mat img=imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
    //To store the keypoints that will be extracted by SIFT
    vector<KeyPoint> keypoints;
    //Detect SIFT keypoints (or feature points)
    detector->detect(img,keypoints);
    //To store the BoW (or BoF) representation of the image
    Mat bowDescriptor;
    //extract BoW (or BoF) descriptor from given image
    bowDE.compute(img,keypoints,bowDescriptor);

    //prepare the yml (some what similar to xml) file
    sprintf(imageTag,"img1");
    //write the new BoF descriptor to the file
    fs1 << imageTag << bowDescriptor;

    //You may use this descriptor for classifying the image.

    //release the file storage
    fs1.release();

}

int main(int argc, char *argv[]){

    return boostt();
}




