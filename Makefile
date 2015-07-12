stitch: stitch.cpp
	export PKG_CONFIG_PATH=/usr/local/Cellar/opencv/2.4.11_1/lib/pkgconfig/; g++ stitch.cpp -g -o bin/stitch `pkg-config opencv --cflags --libs` --std=c++11
ndvi: ndvi.cpp
	export PKG_CONFIG_PATH=/usr/local/Cellar/opencv/2.4.11_1/lib/pkgconfig/; g++ ndvi.cpp -o bin/ndvi `pkg-config opencv --cflags --libs` --std=c++11
