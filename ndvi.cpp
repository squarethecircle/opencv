#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp> 
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/legacy/legacy.hpp" 
#include "opencv2/legacy/compat.hpp" 

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

//Vec3b colormap(double val);


int main(int argc, char** argv)
{
	  if (argc < 2 || argc > 4) {
    printf("Usage:\n\tndvi <input> <output> [scale]\n");
    exit(1);
  }

	//Load filename arguments or use defaults
    const char* photo = argv[1];
	const char* photo_out = argv[2];
	const char* scale_str = (argc == 4) ? argv[3] : "0";
	double scale = 0;
	int equalize = 1;
	if (strcmp(scale_str,"0"))
	{
		scale = atof(scale_str);
		equalize = 0;
	}


   	Mat image = imread(photo);
   	Mat ndvi = Mat(image.size(),CV_32FC1);

	for (int i = 0; i < image.rows; i++) {
	    for (int j = 0; j < image.cols; j++) {
	        Vec3b &intensity = image.at<Vec3b>(i, j);
	            // calculate pixValue

            float b = (float) intensity.val[0];
            float g = (float) intensity.val[1];
            float r = (float) intensity.val[2];
            float val = (r - b)/(r + b);
            if (!equalize) val *= scale;
            ndvi.at<float>(i,j) = val;

     }
	}
	Mat ndvi_flat;
	ndvi.convertTo(ndvi_flat,CV_8U,255.0);
	if (equalize) equalizeHist(ndvi_flat,ndvi_flat);
	Mat cm_ndvi;
	applyColorMap(ndvi_flat,cm_ndvi,COLORMAP_JET);

	imwrite(argv[2],cm_ndvi);
}

/*

Vec3b colormap(float val)
{
	float a = (1-val)/0.25;	//invert and group
	int X = (int) a;	//this is the integer part
	uchar Y = (uchar) 255*(a-X); //fractional part from 0 to 255
	uchar r,g,b;
	switch(X)
	{
	    case 0: r=255;g=Y;b=0;break;
	    case 1: r=255-Y;g=255;b=0;break;
	    case 2: r=0;g=255;b=Y;break;
	    case 3: r=0;g=255-Y;b=255;break;
	    case 4: r=0;g=0;b=255;break;
	}
	Vec3b result;
	result.val[0] = b;
	result.val[1] = g;
	result.val[2] = r;
	return result;


}
*/
