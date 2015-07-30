//

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp> 
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/ocl.hpp>
#include "opencv2/legacy/legacy.hpp" 
#include "opencv2/legacy/compat.hpp" 
#include "opencv2/ocl/ocl.hpp"
#include "akaze/akaze_features.h"


#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <regex>

using namespace std;
using namespace cv;
using namespace cv::ocl;

//ofstream gain_file;


std::string exec(const char* cmd) {
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return "ERROR";
    char buffer[128];
    std::string result = "";
    while(!feof(pipe)) {
    	if(fgets(buffer, 128, pipe) != NULL)
    		result += buffer;
    }
    pclose(pipe);
    return result;
}




double median( Mat channel )
    {
        double m = (channel.rows*channel.cols) / 2;
        int bin = 0;
        double med = -1.0;
 
        int histSize = 256;
        float range[] = { 0, 256 };
        const float* histRange = { range };
        bool uniform = true;
        bool accumulate = false;
        Mat hist;
        calcHist( &channel, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
 
        for ( int i = 0; i < histSize && med < 0.0; ++i )
        {
            bin += floor( hist.at< float >( i ) + 0.5 );
            if ( bin > m && med < 0.0 )
                med = i;
        }
 
        return med;
    }


void printHomography(Mat H)
{
	for (int j = 0; j < 3; j++)
	{
		cout<<" | ";
		for (int k = 0 ; k < 3; k++)
		{
			cout<< H.at<double>(j,k)<<'\t';
		}
		cout<<" |"<<endl;
	}
}


Mat computeHomography( const vector<KeyPoint> & objectKeypoints, const Mat & objectDescriptors,
                    const vector<KeyPoint> & imageKeypoints, const Mat & imageDescriptors,
                    const Mat & color_object_mat, const Mat & color_image_mat, int color)
{


	Mat color_object_mat_float, color_image_mat_float;
   	
   	BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > matches;
    matcher.knnMatch(objectDescriptors,imageDescriptors,matches,2);

	// FLANN interface
	/*
	
	flann::Index tree(imageDescriptors,flann::HierarchicalClusteringIndexParams(32,cvflann::FLANN_CENTERS_KMEANSPP,4,100),cvflann::FLANN_DIST_HAMMING ); 
	Mat indices, dists;
	vector< vector<DMatch> > matches;
	tree.knnSearch(objectDescriptors, indices, dists, 2, cv::flann::SearchParams());

	*/


    vector<DMatch> good_matches;
    //vector< std::pair<int,float> > match_ratios;
    for( int i = 0; i < objectDescriptors.rows; i++ )
	 { 
	 	if (matches[i].size() == 2)
	 	{
	 		//match_ratios.push_back(std::pair<int,float>(i,matches[i][0].distance/matches[i][1].distance));
	 		if (matches[i][0].distance > matches[i][1].distance*0.6) continue;
	 	}
	 		good_matches.push_back( matches[i][0]); 

	 }

	/*
	if (good_matches.size() < 4)
	 {
	 	good_matches.clear();
	 	sort(match_ratios.begin(),match_ratios.end());
	 	for (int i = 0; i < 6;i++)
	 	{
	 		good_matches.push_back(matches[match_ratios[i].first][0]);
	 	}

	 }
	 */
	// FLANN interface
	 /*
	for( int i = 0; i < objectDescriptors.rows; i++ )
	 { 
	 	
	 	if (dists.at<float>(i,0) > dists.at<float>(i,1)*0.6) continue;
	 	
	 		DMatch match(i,indices.at<int>(i,0),dists.at<float>(i,0));
	 		//printf("i: %d\nquery:%d\n",i,match.queryIdx);
	 		good_matches.push_back(match); 

	 }
	 
	*/
	 cout << "Good matches: " << good_matches.size() << "\n";
	 std::vector< Point2f > obj;
	 std::vector< Point2f > img;
	 
	Vec3f obj_sum(0,0,0);
	Vec3f img_sum(0,0,0);
	color_object_mat.convertTo(color_object_mat_float,CV_32FC3,(1.0/255));
	color_image_mat.convertTo(color_image_mat_float,CV_32FC3,(1.0/255));
	Mat homography;
	if (good_matches.size() < 5)
	{
		return homography;
	}

	for( int i = 0; i < good_matches.size(); i++ )
	 {
	 	 Point2f obj_pt = objectKeypoints[ good_matches[i].queryIdx ].pt;
	 	 Point2f img_pt = imageKeypoints[ good_matches[i].trainIdx ].pt;
	 	 obj_sum += color_object_mat_float.at<Vec3f>((int)obj_pt.y, (int)obj_pt.x);
	 	 img_sum += color_image_mat_float.at<Vec3f>((int)img_pt.y, (int)img_pt.x);
		 obj.push_back(obj_pt);
		 img.push_back(img_pt);
	 }

	 float bscale = img_sum[0]/obj_sum[0];
	 float gscale = img_sum[1]/obj_sum[1];
	 float rscale = img_sum[2]/obj_sum[2];
	 //cout << "BLUE: " <<  (bscale) << endl;
	 //cout << "GREEN: " << (gscale) << endl;
	 //cout << "RED: " << (rscale)  << endl;

	 //gain_file << rscale << ", " << bscale << ", " << gscale << endl;
	 if (color == 1)
	 {
		 vector<Mat> channels(3);
		 split(color_object_mat,channels);
		 channels[0] *= bscale;
		 channels[1] *= gscale;
		 channels[2] *= rscale;
		 merge(channels, color_object_mat);
	}
	homography = estimateRigidTransform(obj, img,false);

	
	Mat row = Mat::zeros(1,3,homography.type());
	row.at<double>(0,2) = 1;
	homography.push_back(row);
	
	//homography = findHomography(obj,img,CV_RANSAC);
	/*
	for( int i = 0; i < 4; i++ )
	    {
		   	double x = src_corners[i].x, y = src_corners[i].y;
		   	vector<double> init_pts;
		   	init_pts.push_back(x);
		   	init_pts.push_back(y);
		   	init_pts.push_back(1);

		    double Z = 1./homography.row(2).dot(init_pts);
	        double X = (homography.row(0).dot(init_pts))*Z;
	        double Y = (homography.row(1).dot(init_pts))*Z;
	        dst_corners[i] = Point(floor(X+0.5), floor(Y+0.5));
	        printf("(%d,%d) to (%d,%d)\n",(int)x,(int)y,(int)(X+0.5),(int)(Y+0.5));
	    }
	*/
	return homography;
	/*
	std::vector< Point2f > obj2;
	std::vector< Point2f > img2;
    obj2.push_back(src_corners[0]);
    obj2.push_back(src_corners[1]);
    obj2.push_back(src_corners[2]);
    img2.push_back(dst_corners[0]);
    img2.push_back(dst_corners[1]);

    float dy = dst_corners[1].y-dst_corners[0].y;
    float dx = dst_corners[1].x-dst_corners[0].x;
    float dist = norm(Point2f(dy,dx));
    float ratio = dist/color_object_mat.cols;
    Point2f corner = src_corners[2]*ratio;
    float theta = atan2f(dy,dx);
    Point2f corner2;
	corner2.x =corner.x*cos(theta) + corner.y*sin(theta);
	corner2.y=(-1)*corner.x*sin(theta) + corner.y*cos(theta);
	img2.push_back(corner2);
    
    Mat homography2 = getAffineTransform(obj2,img2);
    homography2.push_back(row);
    	for( int i = 0; i < 4; i++ )
	    {
		   	double x = src_corners[i].x, y = src_corners[i].y;
		   	vector<double> init_pts;
		   	init_pts.push_back(x);
		   	init_pts.push_back(y);
		   	init_pts.push_back(1);

		    double Z = 1./homography2.row(2).dot(init_pts);
	        double X = (homography2.row(0).dot(init_pts))*Z;
	        double Y = (homography2.row(1).dot(init_pts))*Z;
	        dst_corners[i] = Point(floor(X+0.5), floor(Y+0.5));
	    }
    return homography2;
    */
}

int main(int argc, char** argv)
{
  if (argc < 4) {
    printf("Usage:\n\tstitch <input1><other inputs...> <output>\n");
    exit(1);
  }

  	initModule_nonfree();
  	//gain_file.open ("gainout.txt");


    const char* object_filename = argv[2];
    const char* scene_filename = argv[1];
    int filename_index = 3;

	string output_filename;

	output_filename.assign(argv[argc-1]);

	
	Mat color_image_mat = imread(scene_filename);
	/*
	string image_string = string(scene_filename);
	string init_grep_output = exec(("strings "+image_string+" | grep gain").c_str());
	smatch init_red_match;
	smatch init_blue_match;
	smatch init_exp_match;
	regex red_gain_regex("gain_r=(.*?)\\s");
	regex blue_gain_regex("gain_b=(.*?)\\s");
	regex exp_gain_regex("exp=(.*?)\\s");
	regex_search (init_grep_output,init_red_match,red_gain_regex);
	regex_search (init_grep_output,init_blue_match,blue_gain_regex);
	regex_search (init_grep_output,init_exp_match,exp_gain_regex);
	const double init_red_gain = stod(init_red_match[1].str());
	const double init_blue_gain = stod(init_blue_match[1].str());
	const double init_exp_gain = stod(init_exp_match[1].str());
	const double init_green_gain = 2/(init_red_gain+init_blue_gain);

	double rscale;
	double bscale;
	double gscale;
	*/
	Mat color_object_mat = imread(object_filename);
	string object_string = string(object_filename);

	
	/*
	string grep_output = exec(("strings "+object_string+" | grep gain").c_str());
	cout << grep_output << endl;
	smatch red_match;
	smatch blue_match;
	smatch exp_match;
	regex_search (grep_output,red_match,red_gain_regex);
	regex_search (grep_output,blue_match,blue_gain_regex);
	regex_search (grep_output,exp_match,exp_gain_regex);
	cout << red_match[1]<<endl;
	cout << blue_match[1]<<endl;
	cout << exp_match[1]<<endl;
	double red_gain = stod(red_match[1].str());
	double blue_gain = stod(blue_match[1].str());
	double exp_gain = stod(exp_match[1].str());
	double green_gain = 2/(red_gain+blue_gain);


	//gain_file << init_red_gain/red_gain << ", " << init_blue_gain/blue_gain << ", " << init_exp_gain/exp_gain << endl;

	double exp_factor = (init_exp_gain/exp_gain);
	double r_factor = init_red_gain/red_gain;
	double b_factor = init_blue_gain/blue_gain;
	double g_factor = init_green_gain/green_gain;



	vector<Mat> channels(3);
	split(color_object_mat,channels);
	//channels[0] *= 1.043*pow(b_factor,1.222)*pow(exp_factor,1.036);
	//channels[1] *= 1.018*pow(g_factor,0.9181)*pow(exp_factor,0.8773);
	//channels[2] *= 0.9937*pow(r_factor,1.267)*pow(exp_factor,0.3947);
	merge(channels, color_object_mat);
	*/


	    if( !color_object_mat.data || !color_image_mat.data )
	    {
	          	
	        fprintf( stderr, "Can not load image\n");
	        exit(-1);
	    }

	Mat color_image_mat_mask = Mat(color_image_mat.rows, color_image_mat.cols, CV_8UC1, Scalar(1));


	 double t0 = (double)getTickCount();
	 Point past_temp_array[4];
	 Point temp_array[4];
	 Mat H_current = Mat::eye(3,3, CV_64F);
	 Mat color_past_raw, past_raw;

    while(filename_index < argc)
    {
    	double t1 = (double)getTickCount();
    	Mat color_object_copy = color_object_mat;


    	Mat object_mat, image_mat;

		cvtColor( color_object_mat, object_mat, CV_BGR2GRAY );
		cvtColor( color_image_mat, image_mat, CV_BGR2GRAY );

		Mat object_sobel;


		Sobel(object_mat,object_sobel,CV_32F,1,1);
		float blur_factor = sum(abs(object_sobel))[0];
		cout << "Blur factor: " << blur_factor << endl;
		
    	if (blur_factor < 6500000)
		{
			std::cout<<"Too blurry?"<<endl;
			/*
			cout << "Stitching " << argv[filename_index] << "\n";
			string new_file = string(argv[filename_index++]);
			color_past_raw = color_object_copy;
	    	color_object_mat = imread(new_file);
	    	continue;
	    	*/
				
		}
		if (!color_past_raw.empty()) cvtColor( color_past_raw, past_raw, CV_BGR2GRAY );



		//equalizeHist( object_mat, object_mat );
		//equalizeHist( image_mat, image_mat );
		//GaussianBlur(object_mat, object_mat,Size(3,3),0,0);
		//GaussianBlur(image_mat, image_mat,Size(3,3),0,0);

		Mat object_edges, image_edges;

		//Canny( object_mat, object_mat, 90, 100, 3 );
		//Canny( image_mat, image_mat, 90, 100, 3 );















		Mat object_small_mat;
		Mat image_small_mat;







		
			
	    int i;

   		vector<KeyPoint> keypoints_object,keypoints_image,tempk_obj,tempk_img;
   		Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");
   		//Ptr<FeatureDetector> detector(new DynamicAdaptedFeatureDetector (new SurfAdjuster(400,2,1000),10000, 15000, 5));
   		/*
   		Ptr<FeatureDetector> detector(new DynamicAdaptedFeatureDetector (new FastAdjuster(20,true),10000,15000,10));
   		GridAdaptedFeatureDetector grid_detector(detector,15000,8,8);
	    grid_detector.detect(object_mat,keypoints_object);
	    grid_detector.detect(image_mat,keypoints_image,color_image_mat_mask);
	*/
	    //Ptr<FeatureDetector> detector = FeatureDetector::create("FAST");
   		//Ptr<FeatureDetector> detector(new DynamicAdaptedFeatureDetector (new SurfAdjuster(400,2,1000),10000, 15000, 5));
	    //Ptr<FeatureDetector> detector(new DynamicAdaptedFeatureDetector (new FastAdjuster(20,true),40000,50000,10));
   		GridAdaptedFeatureDetector grid_detector(detector,20000,4,4);
	    Mat descriptors_object,descriptors_image,tempd_obj,tempd_img;
	    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("BRISK");
	    Point2f src_corners_init[4] = {Point2f(0,0), Point2f(color_object_mat.cols,0), Point2f(color_object_mat.cols, color_object_mat.rows), Point2f(0, color_object_mat.rows)};
	    vector<Point2f> dst_corners;

	    vector<Point2f> temp_dst_corners(src_corners_init,src_corners_init+4);

	    if (!past_raw.empty())
	    {
	   		grid_detector.detect(object_mat,tempk_obj);
		    grid_detector.detect(past_raw,tempk_img);
		    extractor->compute(object_mat,tempk_obj,tempd_obj);
		    extractor->compute(past_raw,tempk_img,tempd_img);
		    Point2f temp_src_corners_init[4] = {Point2f(0,0), Point2f(color_object_mat.cols,0), Point2f(color_object_mat.cols, color_object_mat.rows), Point2f(0, color_object_mat.rows)};
		    Point2f temp_img_corners[4] = {Point2f(0,0), Point2f(color_image_mat.cols,0), Point2f(color_image_mat.cols, color_image_mat.rows), Point2f(0, color_image_mat.rows)};

		    vector<Point2f> temp_src_corners(temp_src_corners_init,temp_src_corners_init+4);

			Mat H_new = computeHomography(tempk_obj,tempd_obj,tempk_img,tempd_img,color_object_mat,past_raw,0);
			perspectiveTransform(temp_src_corners,temp_dst_corners,H_new*H_current);

			//printHomography(H_new);
			//Point temp_img_corners[4] = {Point(0,0), Point(past_raw.cols,0), Point(past_raw.cols, past_raw.rows), Point(0, past_raw.rows)};

			for( i = 0; i < 4; i++ )
		    {
		    	printf("(%d,%d) to (%d,%d)\n",(int)temp_src_corners[i].x,(int)temp_src_corners[i].y,(int)(temp_dst_corners[i].x+0.5),(int)(temp_dst_corners[i].y+0.5));
		    }

			vector<Point2f> temp_cornerPts;

			Rect temp_mbr = boundingRect(temp_dst_corners);
						cout << "tempwidth, tempheight: " << temp_mbr.width << "," <<temp_mbr.height << endl;

			Mat temp_H_shift = Mat::eye(3,3, CV_64F);

			temp_H_shift.at<double>(0,2) = -temp_mbr.x;
			temp_H_shift.at<double>(1,2) = -temp_mbr.y;
			Mat temp_H_combo = temp_H_shift*H_new*H_current;
			printHomography(H_current);
			printHomography(temp_H_shift);
			printHomography(H_new);
			printHomography(temp_H_combo);

			/*
			temp_H_combo.at<double>(0,1) = abs(temp_H_combo.at<double>(0,1));
			temp_H_combo.at<double>(1,1) = abs(temp_H_combo.at<double>(1,1));
			temp_H_combo.at<double>(0,2) = abs(temp_H_combo.at<double>(0,2));
			temp_H_combo.at<double>(1,2) = abs(temp_H_combo.at<double>(1,2));
			*/
			perspectiveTransform(temp_src_corners,temp_dst_corners,temp_H_combo);


			for( i = 0; i < 4; i++ )
		    {
		    	printf("(%d,%d) to (%d,%d)\n",(int)temp_src_corners[i].x,(int)temp_src_corners[i].y,(int)(temp_dst_corners[i].x+0.5),(int)(temp_dst_corners[i].y+0.5));
		    }




			/*

			for( i = 0; i < 4; i++ )
		    {
			   	double tempx = temp_src_corners[i].x, tempy = temp_src_corners[i].y;
			   	vector<double> temp_init_pts;
			   	temp_init_pts.push_back(tempx);
			   	temp_init_pts.push_back(tempy);
			   	temp_init_pts.push_back(1);

		        double tempX = (temp_H_combo.row(0).dot(temp_init_pts));
		        double tempY = (temp_H_combo.row(1).dot(temp_init_pts));
		        temp_dst_corners[i] = Point(floor(tempX+0.5), floor(tempY+0.5));
		       	printf("(%d,%d) to (%d,%d)\n",(int)tempx,(int)tempy,(int)(tempX+0.5),(int)(tempY+0.5));

		    }
		    */

			

		
			//temp_cornerPts.insert(temp_cornerPts.begin(),temp_img_corners,temp_img_corners + 4);
			temp_mbr = boundingRect(temp_dst_corners);



			Mat temp = Mat::zeros(Size(temp_mbr.width,temp_mbr.height),CV_8UC3);
			warpPerspective(color_object_mat, temp, temp_H_combo, Size(temp_mbr.width,temp_mbr.height), CV_INTER_CUBIC, BORDER_TRANSPARENT);

		
			/*
			Mat gray_temp;
			cvtColor(temp,gray_temp,CV_BGR2GRAY);
			vector< vector<Point> > contours; // Vector for storing contour
		    vector<Vec4i> hierarchy;
		 	int largest_area=0;
			int largest_contour_index=0;
			Rect bounding_rect;
		    
		    findContours(gray_temp, contours, hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE ); // Find the contours in the image
		   
		     for( int i = 0; i< contours.size(); i++ ) // iterate through each contour. 
		      {
		       double a=contourArea( contours[i],false);  //  Find the area of contour
		       if(a>largest_area){
		       largest_area=a;
		       largest_contour_index=i;                //Store the index of largest contour
		       }
		      }
			*/
			color_object_mat = temp;
			imwrite("wrecked.png",color_object_mat);
			imwrite("wrecker.png",color_image_mat);
			cvtColor(color_object_mat,object_mat,CV_BGR2GRAY);

		}




		grid_detector.detect(object_mat,keypoints_object);
	    grid_detector.detect(image_mat,keypoints_image);






	    extractor->compute(object_mat,keypoints_object,descriptors_object);
	    printf("Object Descriptors: %d\n", descriptors_object.rows);
	    extractor->compute(image_mat,keypoints_image,descriptors_image);
	    printf("Image Descriptors: %d\n", descriptors_image.rows);
	    double t2 = (double)getTickCount();
		printf( "Extraction time = %gs\n", (t2-t1) /(getTickFrequency()));


	    


		Mat H = computeHomography( keypoints_object, descriptors_object, keypoints_image,
	        descriptors_image,color_object_mat,color_image_mat,0);

	    if(!H.empty())
	    {
			printHomography(H);

	   	}

		else
		{
			std::cout<<"registration match failed hard - are you sure these images overlap?"<<endl;
			cout << "Stitching " << argv[filename_index] << "\n";
			string new_file = string(argv[filename_index++]);
			color_past_raw = color_object_copy;
	    	color_object_mat = imread(new_file);
	    	continue;
				
		}


		Point2f img_corners[4] = {Point2f(0,0), Point2f(color_image_mat.cols,0), Point2f(color_image_mat.cols, color_image_mat.rows), Point2f(0, color_image_mat.rows)};
		
		perspectiveTransform(temp_dst_corners,dst_corners,H);


		vector<Point2f> cornerPts(img_corners, img_corners + 4);
		cornerPts.insert(cornerPts.end(),dst_corners.begin(),dst_corners.end());
		Rect mbr = boundingRect(cornerPts);

		cout<<"minimum bounding rectangle: {x: "<<mbr.x<<"  y: "<<mbr.y<<"  width: "<<mbr.width<<"  height: "<<mbr.height<<endl;

		Mat H_shift = Mat::eye(3,3, CV_64F);
		Mat H_combo;

		int x_offset = mbr.x;
		int y_offset = mbr.y;
		H_shift.at<double>(0,2) = -mbr.x;
		H_shift.at<double>(1,2) = -mbr.y;
		cout<<endl<<"shift transformation: "<<endl;
		printHomography(H_shift);
		H_combo = H_shift * H;
		H_current = H_current * H_combo;

		cout<<endl<<"combined transformation: "<<endl;
		printHomography(H_combo);
		cout <<endl;

		/*
		for( i = 0; i < 4; i++ )
	    {
		   	double x = src_corners[i].x, y = src_corners[i].y;
		   	vector<double> init_pts;
		   	init_pts.push_back(x);
		   	init_pts.push_back(y);
		   	init_pts.push_back(1);

		    double Z = 1./H_combo.row(2).dot(init_pts);
	        double X = (H_combo.row(0).dot(init_pts))*Z;
	        double Y = (H_combo.row(1).dot(init_pts))*Z;
	        dst_corners[i] = Point(floor(X+0.5), floor(Y+0.5));
	    }
	    */
	    perspectiveTransform(temp_dst_corners,dst_corners,H_combo);
		

		
		cornerPts.clear();
		cornerPts.insert(cornerPts.end(),img_corners,img_corners + 4);
		cornerPts.insert(cornerPts.end(),dst_corners.begin(),dst_corners.begin() + 4);
		mbr = boundingRect(cornerPts);

		
		cout<<"minimum bounding rectangle: {x: "<<mbr.x<<"  y: "<<mbr.y<<"  width: "<<mbr.width<<"  height: "<<mbr.height<<endl;

		

		Mat rectifiedImage2 = Mat::zeros(Size(mbr.width,mbr.height),CV_8UC3);
		warpPerspective(color_object_mat, rectifiedImage2, H_combo, Size(mbr.width,mbr.height), CV_INTER_CUBIC, BORDER_TRANSPARENT);

		for (int i = 0; i < rectifiedImage2.rows; i++)
		{
			for (int j = 0; j < rectifiedImage2.cols; j++)
			{
				if (pointPolygonTest(dst_corners,Point2f(j,i),true) < 10)
				{
					Vec3b &colorVals = rectifiedImage2.at<Vec3b>(i,j);
					colorVals.val[0] = 0;
					colorVals.val[1] = 0;
					colorVals.val[2] = 255;
				}
			}
		}

		imwrite("rectified.png",rectifiedImage2);

		cout<<"x offset: "<<x_offset<<"   y offset: "<<y_offset<<endl;


		int x1, y1, x2, y2, w, h;

		int x0,y0,big_width, big_height;

		if(y_offset > 0)
		{
			y2 = 0;
			y1 = abs(y_offset);
			cout<<"image 2 on top"<<endl;
			h = rectifiedImage2.rows + y1;
		}
		else
		{
			y2 = abs(y_offset);
			y1 = 0;
			cout<<"image 1 on top"<<endl;
			h = max(color_image_mat.rows + y2, mbr.height);
		}

		if(x_offset >0) 
		{
			x2 = 0;
			x1 = abs(x_offset);

			cout<<"image 2 on left"<<endl;
			w = rectifiedImage2.cols + x1;
		}
		else
		{
			x2 = abs(x_offset);
			x1 = 0;
			cout<<"image 1 on left"<<endl;
			w = max(color_image_mat.cols + x2, mbr.width);
		}
		
		Mat rectContour;
		cvtColor(rectifiedImage2, rectContour, CV_BGR2GRAY);
		vector<vector<Point> > contours;
		// Finds contours

		// Calculates the bounding rect of the largest area contour
		Rect rectRect = boundingRect(dst_corners);
		

		x0 = x1 + rectRect.x - rectRect.width/2;
		y0 = y1 + rectRect.y - rectRect.height/2;
		big_width = (int)rectRect.width*3;
		big_height = (int)rectRect.height*3;

		if (x0 < 0) x0 = 0;
		if (y0 < 0) y0 = 0;
		int extra_width = x0 + big_width - w;
		int extra_height = y0 + big_height - h;

		if (extra_width > 0) big_width -= extra_width;
		if (extra_height > 0) big_height -= extra_height;



		Size outSize = Size(w,h);
		cout<<"outsize: "<<outSize.width<<" x "<<outSize.height<<endl;
		cout<<"rect2size: "<<rectifiedImage2.cols<<" x "<<rectifiedImage2.rows<<endl;
		cout<<"colormatsize: "<<color_image_mat.cols<<" x"<<color_image_mat.rows<<endl;

		Mat comboImage = Mat::zeros(outSize,CV_8UC3);
		Mat rectifiedMask;

		color_image_mat.copyTo(comboImage(Rect(x2,y2,color_image_mat.cols, color_image_mat.rows)));

		//inRange(rectifiedImage2,Scalar(10,10,10),Scalar(255,255,255),rectifiedMask);
		vector<Mat> r_chan;
		split(rectifiedImage2,r_chan);
		bitwise_not(r_chan[2],r_chan[2]);
		rectifiedImage2.copyTo(comboImage(Rect(x1,y1,rectifiedImage2.cols, rectifiedImage2.rows)),r_chan[2]);

		cout<<x1<<", "<<x1+rectifiedImage2.cols<<endl;
		cout<<y1<<", "<<y1+rectifiedImage2.rows<<endl;


		if (filename_index < argc - 1)
		{	
			color_image_mat = comboImage;
			
			color_image_mat_mask = Mat(comboImage.rows, comboImage.cols, CV_8UC1, Scalar(0));


						//cvtColor( rectifiedImage2, rectgray, CV_BGR2GRAY );
			//rectgray.copyTo(color_image_mat_mask(Rect(x1,y1,rectgray.cols, rectgray.rows)));
			cout << "x0,y0,x0+big_width,y0+big_height:  " << x0 << " " << y0  << " " << x0+big_width << " " << y0+big_height;
			Point roi_poly_int[4] = {Point(x0,y0),Point(x0+big_width,y0),Point(x0+big_width,y0+big_height),Point(x0,y0+big_height)};
			fillConvexPoly(color_image_mat_mask, roi_poly_int, 4, 255, 8, 0);
			imwrite("lastmask.png",color_image_mat_mask);
			imwrite("last.png", comboImage );


			//imwrite("last.jpg", color_object_mat_mask );
			//imwrite("lastrec.jpg", rectifiedImage2 );
			/*

			copy(temp_array,temp_array+4,past_temp_array);

			temp_array[0] = Point(x1,y1);
			temp_array[1] = Point(x1,y1+rectifiedImage2.rows);
			temp_array[2] = Point(x1+rectifiedImage2.cols,y1);
			temp_array[3] = Point(x1+rectifiedImage2.cols,y1+rectifiedImage2.rows);
			vector<Point> roi_vertices(&temp_array[0],&temp_array[0]+4);
			//roi_vertices.insert(roi_vertices.end(),past_temp_array,past_temp_array+4);
			RotatedRect mask_rect = minAreaRect(roi_vertices);
			Point2f roi_poly[4];
			mask_rect.points(roi_poly);

			Point roi_poly_int[4] = {Point(roi_poly[0].x,(int)roi_poly[0].y),Point(roi_poly[1].x,(int)roi_poly[1].y),Point(roi_poly[2].x,(int)roi_poly[2].y),Point(roi_poly[3].x,(int)roi_poly[3].y)};
			
			fillConvexPoly(color_object_mat_mask, roi_poly_int, 4, 100, 8, 0);
			*/
			

			cout << "Stitching " << argv[filename_index] << "\n";
			string new_file = string(argv[filename_index++]);
			color_past_raw = color_object_copy;
	    	color_object_mat = imread(new_file);
	    	/*
	    	string grep_output = exec(("strings "+new_file+" | grep gain").c_str());
	    	cout << grep_output << endl;
	    	regex_search (grep_output,red_match,red_gain_regex);
	    	regex_search (grep_output,blue_match,blue_gain_regex);
	    	regex_search (grep_output,exp_match,exp_gain_regex);
    		cout << red_match[1]<<endl;
    		cout << blue_match[1]<<endl;
    		cout << exp_match[1]<<endl;
    		red_gain = stod(red_match[1].str());
    		blue_gain = stod(blue_match[1].str());
    		exp_gain = stod(exp_match[1].str());
    		r_factor = init_red_gain/red_gain;
			b_factor = init_blue_gain/blue_gain;
			g_factor = init_green_gain/green_gain;



			//gain_file << init_red_gain/red_gain << ", " << init_blue_gain/blue_gain << ", " << init_exp_gain/exp_gain << endl;

    		exp_factor = (init_exp_gain/exp_gain);
    		vector<Mat> channels(3);
			split(color_object_mat,channels);
			//channels[0] *= 1.043*pow(b_factor,1.222)*pow(exp_factor,1.036);
			//channels[1] *= 1.018*pow(g_factor,0.9181)*pow(exp_factor,0.8773);
			//channels[2] *= 0.9937*pow(r_factor,1.267)*pow(exp_factor,0.3947);
			merge(channels, color_object_mat);
			*/


	    	if(!color_object_mat.data )
	    {
	        fprintf( stderr, "Can not load image\n");
	        exit(-1);
	    }
	    }

	   	else
	   	{
		   	imwrite(output_filename.c_str(), comboImage );
		   	//gain_file.close();
			double t3 = (double)getTickCount();
			printf( "Total elapsed time = %gs\n", (t3-t0) /(getTickFrequency()));
			break;

	   	}


	}
	return 0;


}
