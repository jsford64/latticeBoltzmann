all:
	g++ latBolt.cpp -O3 -o latBolt -lfmt -I/usr/include/ -I/usr/include/eigen3/ -I/usr/include/eigen3/unsupported

clean:
	rm *.csv latBolt
