stitch: stitch.cpp
	export PKG_CONFIG_PATH=/usr/local/Cellar/opencv/2.4.11_1/lib/pkgconfig/; g++ $^ -g -o bin/$@ `pkg-config opencv --cflags --libs`
ndvi: ndvi.cpp
	export PKG_CONFIG_PATH=/usr/local/Cellar/opencv/2.4.11_1/lib/pkgconfig/; g++ $^ -o bin/$@ `pkg-config opencv --cflags --libs`
scale: scale.cpp
	export PKG_CONFIG_PATH=/usr/local/Cellar/opencv/2.4.11_1/lib/pkgconfig/; g++ $^ -g -o bin/$@ `pkg-config opencv --cflags --libs`
