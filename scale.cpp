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
#include <math.h>

using namespace std;
using namespace cv;



int main(int argc, char** argv)
{
	  if (argc < 2 || argc > 4) {
    printf("Usage:\n\tscale <input> <output> [scale]\n");
    exit(1);
  }

	//Load filename arguments or use defaults
    const char* photo = argv[1];
	const char* photo_out = argv[2];
	const char* scale_str = (argc == 4) ? argv[3] : "0";
	double scale = 0;
	int equalize = 1;



   	Mat image = imread(photo);
   	vector<Mat> channels(3);
   	image *= 0.973;
   	split(image, channels);
	channels[0] *= pow(1.104,2);
	channels[2] *= pow(0.96498,2);

	Mat output;
	merge(channels,output);



	imwrite(argv[2],output);
}