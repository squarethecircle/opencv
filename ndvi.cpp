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

Vec3b colormap(double val);


int main(int argc, char** argv)
{
	  if (argc < 2 || argc > 3) {
    printf("Usage:\n\tndvi <input> <output>\n");
    exit(1);
  }

	//Load filename arguments or use defaults
    const char* photo = argv[1];
	const char* photo_out = argv[2];
   	Mat image = imread(photo);

	for (int i = 0; i < image.cols; i++) {
	    for (int j = 0; j < image.rows; j++) {
	        Vec3b &intensity = image.at<Vec3b>(j, i);
	            // calculate pixValue

	            double b = (double) intensity.val[0]+10;
	            double g = (double) intensity.val[1]+10;
	            double r = (double) intensity.val[2]+10;
	            double ndvi = (r - b)/(r + b);
	           	Vec3b colormapped = colormap(4*ndvi);
	           	intensity.val[0] = colormapped.val[0];
	           	intensity.val[1] = colormapped.val[1];
	           	intensity.val[2] = colormapped.val[2];
	     }
	}
	imwrite(argv[2],image);
}


Vec3b colormap(double val)
{
	double a = (1-val)/0.25;	//invert and group
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
