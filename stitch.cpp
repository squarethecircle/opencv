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


#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::ocl;





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
                    Point src_corners[4], Point dst_corners[4])
{



   	BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > matches;
    matcher.knnMatch(objectDescriptors,imageDescriptors,matches,2);


	/* FLANN interface
	flann::Index tree(imageDescriptors,flann::HierarchicalClusteringIndexParams(32,cvflann::FLANN_CENTERS_KMEANSPP,4,100),cvflann::FLANN_DIST_HAMMING ); 
	Mat indices, dists;  
	vector< vector<DMatch> > matches;
	tree.knnSearch(objectDescriptors, indices, dists, 2, cv::flann::SearchParams());

	*/

    vector<DMatch> good_matches;
    vector< std::pair<int,float> > match_ratios;

    for( int i = 0; i < objectDescriptors.rows; i++ )
	 { 
	 	if (matches[i].size() == 2)
	 	{
	 		match_ratios.push_back(std::pair<int,float>(i,matches[i][0].distance/matches[i][1].distance));
	 		if (matches[i][0].distance > matches[i][1].distance*0.6) continue;
	 	}
	 		good_matches.push_back( matches[i][0]); 

	 }
	if (good_matches.size() < 4)
	 {
	 	good_matches.clear();
	 	sort(match_ratios.begin(),match_ratios.end());
	 	for (int i = 0; i < 6;i++)
	 	{
	 		good_matches.push_back(matches[match_ratios[i].first][0]);
	 	}

	 }
	/*  FLANN interface
	for( int i = 0; i < objectDescriptors.rows; i++ )
	 { 
	 	
	 	//if (dists.at<float>(i,0) > dists.at<float>(i,1)*0.6) continue;
	 	
	 		//DMatch match(i,indices.at<int>(i,0),dists.at<float>(i,0));
	 		//printf("i: %d\nquery:%d\n",i,match.queryIdx);
	 		good_matches.push_back(match); 

	 }
	*/
	 cout << "Good matches: " << good_matches.size() << "\n";
	 std::vector< Point2f > obj;
	 std::vector< Point2f > img;
	 
	for( int i = 0; i < good_matches.size(); i++ )
	 {
		 obj.push_back( objectKeypoints[ good_matches[i].queryIdx ].pt );
		 img.push_back( imageKeypoints[ good_matches[i].trainIdx ].pt );
	 }

	Mat homography = findHomography(obj,img, CV_LMEDS);
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
	    }
    return homography;
}

int main(int argc, char** argv)
{
  if (argc < 4) {
    printf("Usage:\n\tstitch <input1><other inputs...> <output>\n");
    exit(1);
  }

  	initModule_nonfree();


    const char* object_filename = argv[1];
    const char* scene_filename = argv[2];
    int filename_index = 3;

	string output_filename;

	output_filename.assign(argv[argc-1]);

	Mat color_object_mat = imread(object_filename);


	Mat color_image_mat = imread(scene_filename);


	    if( !color_object_mat.data || !color_image_mat.data )
	    {
	          	
	        fprintf( stderr, "Can not load image\n");
	        exit(-1);
	    }

	Mat color_object_mat_mask = Mat(color_object_mat.rows, color_object_mat.cols, CV_8UC1, Scalar(1));




	 static CvScalar colors[] = 
	    {
	        {{0,0,255}},
	        {{0,128,255}},
	        {{0,255,255}},
	        {{0,255,0}},
	        {{255,128,0}},
	        {{255,255,0}},
	        {{255,0,0}},
	        {{255,0,255}},
	        {{255,255,255}}
	    };
	 double t0 = (double)getTickCount();
	 Point past_temp_array[4];
	 Point temp_array[4];

    while(filename_index < argc)
    {
    	double t1 = (double)getTickCount();


    	Mat object_mat, image_mat;





		vector<Mat> channels_obj;
		vector<Mat> channels_img;

		split(color_object_mat,channels_obj);
		split(color_image_mat,channels_img);

		double med_0 = median(channels_obj[0]) - median(channels_img[0]);
		double med_1 = median(channels_obj[1]) - median(channels_img[1]);
		double med_2 = median(channels_obj[2]) - median(channels_img[2]);

		add(Scalar(0),channels_img[0],channels_img[0]);
		add(Scalar(0),channels_img[1],channels_img[1]);
		add(Scalar(0),channels_img[2],channels_img[2]);




		merge(channels_obj, color_object_mat);
		merge(channels_img, color_image_mat);


		cvtColor( color_object_mat, object_mat, CV_BGR2GRAY );
		cvtColor( color_image_mat, image_mat, CV_BGR2GRAY );



		equalizeHist( object_mat, object_mat );
		equalizeHist( image_mat, image_mat );
		//GaussianBlur(object_mat, object_mat,Size(3,3),0,0);
		//GaussianBlur(image_mat, image_mat,Size(3,3),0,0);

		Mat object_edges, image_edges;

		//Canny( object_mat, object_mat, 90, 100, 3 );
		//Canny( image_mat, image_mat, 90, 100, 3 );















		Mat object_small_mat;
		Mat image_small_mat;







		
			
	    int i;

   		vector<KeyPoint> keypoints_object,keypoints_image;

   		Ptr<FeatureDetector> detector = FeatureDetector::create("GridSURF");
	    detector->detect(object_mat,keypoints_object,color_object_mat_mask);
	    detector->detect(image_mat,keypoints_image);




		Mat descriptors_object,descriptors_image;




		Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("BRISK");


	    extractor->compute(color_object_mat,keypoints_object,descriptors_object);
	    printf("Object Descriptors: %d\n", descriptors_object.rows);
	    extractor->compute(color_image_mat,keypoints_image,descriptors_image);
	    printf("Image Descriptors: %d\n", descriptors_image.rows);
	    double t2 = (double)getTickCount();
		printf( "Extraction time = %gms\n", (t2-t1) /(getTickFrequency()*1000.));


		Point src_corners[4] = {Point(0,0), Point(color_object_mat.cols,0), Point(color_object_mat.cols, color_object_mat.rows), Point(0, color_object_mat.rows)};
	    Point dst_corners[4];
	    



		Mat H = computeHomography( keypoints_object, descriptors_object, keypoints_image,
	        descriptors_image, src_corners, dst_corners);
	    if(!H.empty())
	    {
			printHomography(H);
	    }

		else
		{
			std::cout<<"registration match failed hard - are you sure these images overlap?"<<endl;
			return 1;
		}


		Point img_corners[4] = {Point(0,0), Point(color_image_mat.cols,0), Point(color_image_mat.cols, color_image_mat.rows), Point(0, color_image_mat.rows)};
		

		vector<Point> cornerPts(img_corners, img_corners + 4);
		cornerPts.insert(cornerPts.end(),dst_corners,dst_corners + 4);
		Rect mbr = boundingRect(cornerPts);

		cout<<"minimum bounding rectangle: {x: "<<mbr.x<<"  y: "<<mbr.y<<"  width: "<<mbr.width<<"  height: "<<mbr.height<<endl;

		Mat H_shift = Mat::eye(3,3, CV_64F);
		Mat H_shift2 = Mat::eye(3,3, CV_64F);
		Mat H_combo;
		Mat H_combo2;
		Mat H_scale = Mat::eye(3,3, CV_64F);
		Mat test;
		Mat test2;
		Mat test3;

		H_scale.at<double>(0,0) = 0.5;
		H_scale.at<double>(1,1) = 0.5;
		H_scale.at<double>(2,2) = 1;

		int x_offset = mbr.x;
		int y_offset = mbr.y;
		H_shift.at<double>(0,2) = -mbr.x;
		H_shift2.at<double>(0,2) = -(mbr.x * 2) + mbr.width/2; 
		H_shift.at<double>(1,2) = -mbr.y;
		H_shift2.at<double>(1,2) = -(mbr.y * 2) + mbr.height/2;
		cout<<endl<<"shift transformation: "<<endl;
		printHomography(H_shift);
		H_combo = H_shift * H;
		H_combo2 = H * H_shift2;
		H_scale = H_combo2 * H_scale;

		cout<<endl<<"combined transformation: "<<endl;
		printHomography(H_combo);
		cout <<endl;


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
		

		
		cornerPts.clear();
		cornerPts.insert(cornerPts.end(),img_corners,img_corners + 4);
		cornerPts.insert(cornerPts.end(),dst_corners,dst_corners + 4);
		mbr = boundingRect(cornerPts);

		
		cout<<"minimum bounding rectangle: {x: "<<mbr.x<<"  y: "<<mbr.y<<"  width: "<<mbr.width<<"  height: "<<mbr.height<<endl;

		

		Mat rectifiedImage2 = Mat(Size(mbr.width,mbr.height),CV_8UC3);
		warpPerspective(color_object_mat, rectifiedImage2, Mat(H_combo), Size(mbr.width,mbr.height), CV_INTER_LINEAR |  CV_WARP_FILL_OUTLIERS);

		
		cout<<"x offset: "<<x_offset<<"   y offset: "<<y_offset<<endl;


		int x1, y1, x2, y2, w, h;

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
			h = max(color_image_mat.cols + y2, mbr.height);
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
			w = max(color_image_mat.rows + x2, mbr.width);
		}
		Size outSize = Size(w,h);
		cout<<"outsize: "<<outSize.width<<" x "<<outSize.height<<endl;
		cout<<"rect2size: "<<rectifiedImage2.cols<<" x "<<rectifiedImage2.rows<<endl;

		Mat comboImage = Mat::zeros(outSize,CV_8UC3);

		rectifiedImage2.copyTo(comboImage(Rect(x1,y1,rectifiedImage2.cols, rectifiedImage2.rows)));

		color_image_mat.copyTo(comboImage(Rect(x2,y2,color_image_mat.cols, color_image_mat.rows)));

		if (filename_index < argc - 1)
		{	
			color_object_mat = comboImage;
			color_object_mat_mask = Mat(color_object_mat.rows, color_object_mat.cols, CV_8UC1, Scalar(0));
			copy(temp_array,temp_array+4,past_temp_array);

			temp_array[0] = Point(x2,y2);
			temp_array[1] = Point(x2,y2+color_image_mat.rows);
			temp_array[2] = Point(x2+color_image_mat.cols,y2);
			temp_array[3] = Point(x2+color_image_mat.cols,y2+color_image_mat.rows);
			vector<Point> roi_vertices(&temp_array[0],&temp_array[0]+4);
			roi_vertices.insert(roi_vertices.end(),past_temp_array,past_temp_array+4);
			RotatedRect mask_rect = minAreaRect(roi_vertices);
			Point2f roi_poly[4];
			mask_rect.points(roi_poly);

			Point roi_poly_int[4] = {Point(roi_poly[0].x,(int)roi_poly[0].y),Point(roi_poly[1].x,(int)roi_poly[1].y),Point(roi_poly[2].x,(int)roi_poly[2].y),Point(roi_poly[3].x,(int)roi_poly[3].y)};

			fillConvexPoly(color_object_mat_mask, roi_poly_int, 4, 255, 8, 0);
			

			cout << "Stitching " << argv[filename_index] << "\n";
	    	color_image_mat = imread(argv[filename_index++]);
	    	if(!color_image_mat.data )
	    {
	        fprintf( stderr, "Can not load image\n");
	        exit(-1);
	    }
	    }

	   	else
	   	{
		   	imwrite(output_filename.c_str(), comboImage );
			double t3 = (double)getTickCount();
			printf( "Total elapsed time = %gms\n", (t3-t0) /(getTickFrequency()*1000.));
			break;

	   	}


	}
	return 0;


}

