
COMMAND = executive

# 'make' runs the test. same as 'make test'.
all: test
	(echo -e "\n\n'./$(COMMAND)' runs the executable.\n\n'make run' uses mpirun -np 3 ./executive test.jpg test2.jpg test3.jpg\n";)

# 'make test' runs the test
test: $(COMMAND)

# 'make example' runs the sample
example: sample
	(echo -e "\n\nRunning sample with test.jpg...\n"; ./sample test.jpg)


# Make the executive
$(COMMAND): $(COMMAND).o ../lib/libCOGLImageReader.so
	mpic++ -o $(COMMAND) $(COMMAND).o ../lib/libCOGLImageReader.so
$(COMMAND).o: $(COMMAND).c++
	mpic++ -c -std=c++11 -I../Packed3DArray -I../ImageReader $(COMMAND).c++

# Make the sample
sample: sample.o ../lib/libCOGLImageReader.so
	mpic++ -o sample sample.o ../lib/libCOGLImageReader.so
sample.o: sample.c++
	mpic++ -c -std=c++11 -I../Packed3DArray -I../ImageReader sample.c++

# Make the ImageReader.so
../lib/libCOGLImageReader.so: ../ImageReader/ImageReader.h ../ImageReader/ImageReader.c++ ../Packed3DArray/Packed3DArray.h
	(cd ../ImageReader; make)

run:
	mpirun -np 3 ./executive test.jpg test2.jpg test3.jpg

clean:
	rm -rf *.o $(COMMAND) sample
