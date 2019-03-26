// executive.c++

#include <mpi.h>
#include <cmath> //For abs()
#include "ImageReader.h"

//Toggle verbose output
const bool VERBOSE = true;

//MPI Datatypes
MPI_Datatype* MPI_HISTOGRAM;

//Histogram struct, because semantic convenience
typedef struct {
	double** channels; //3x256 double array, respective of R,G,B channels
} Histogram;

//histogram constructor
Histogram* initHistogram()
{
	Histogram* hist = new Histogram;
	hist->channels = new double*[3];
	for (int i=0 ; i < 3 ; i++) hist->channels[i] = new double[256];
	return hist;
}

//Used by computeRGBHistogram to compute the relative frequency of a given color.
//    pa | Packed 3D array for a given array
// value | RGB value to count
//Returns the frequency in the given image
int computeRGBFrequency(const cryph::Packed3DArray<unsigned char>* pa, int value)
{
	int freq = 0;
	for (int r=0 ; r<pa->getDim1() ; r++) 				//Rows
		for (int c=0 ; c<pa->getDim2() ; c++) 				//Columns
			for (int rgb=0 ; rgb<pa->getDim3() ; rgb++) 	//Channels
				if (pa->getDataElement(r, c, rgb) == value) freq++; //Get frequency of given RGB value.
	return freq;
}

//Computes the RGB color histogram channels for a given Histogram struct.
// 		hist | The given Histogram.
// Returns the Histogram with the computed channels.
void computeRGBHistogram(cryph::Packed3DArray<unsigned char>* pa, Histogram* hist)
{
	int size = pa->getDim1() * pa->getDim2();
	for (int chan=0 ; chan < 3 ; chan++)
		for (int rgb=0 ; rgb < 256 ; rgb++) {
			hist->channels[chan][rgb] = (100 * computeRGBFrequency(pa, rgb)) / size; //Calculate as relative % of image
		}
}

//Given an array of color Histograms and the reference Histogram,
//    compute the similarity vector. Compared against itself returns 0 for the sum.
double* computeSimilarityVector(Histogram* referenceHist, Histogram* histograms, int N)
{
	double* similarityVector = new double[N]; //Vector containing similarity proportion with each other image.
	for (int rank=0 ; rank < N; rank++) {
		double sum = 0;
		for (int channel=0 ; channel < 3; channel++)
			for (int rgb=0 ; rgb < 256 ; rgb++)
				sum += std::abs(histograms[rank].channels[channel][rgb] - referenceHist->channels[channel][rgb]);
		similarityVector[rank] += sum;
	}
	return similarityVector;
}

//N = communicatorSize
void do_rank_i_work(int N)
{
	//(For DEBUG output purposes only - only used for debugging-related couts, no MPI calls)
	int rank_debug;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_debug);

	//Dimensions of 3d unsigned char  work assignment
	int workDims[3];

	//Initialize unsigned char array to hold raw image data from a Packed3DArray
	unsigned char* imageData;

	//Initialize Histogram results arrays. Nx3x256 doubles array.
	//Used to store rank i's 3x256 histogram results in histograms[i] (= 3x256 doubles array)
	//This data is all-gathered after the images are scattered.
	Histogram* histograms = new Histogram[N];
	for (int img=0 ; img < N ; img++) {
		histograms[img].channels = new double*[3];
		for (int rgb=0 ; rgb < 3 ; rgb++) histograms[img].channels[rgb] = new double[256];
	}

	//Initialize memory for rank i's similarity vector.
	double* similarities_i = new double[N];

	//Receive the work size from rank 0 (blocking, although doesn't have to be)
	//Then receive work assignment (image data) (blocking)
	std::cout << "\nRank " << rank_debug << " receiving work size..."; //DEBUG
	MPI_Recv(&workDims, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	std::cout << "\nRank " << rank_debug << " receiving image data..."; //DEBUG
	MPI_Recv(imageData, workDims[0] * workDims[1] * workDims[2], MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	//Recreate the pa with the raw data in the initial buffer
	cryph::Packed3DArray<unsigned char>* image_i = new cryph::Packed3DArray<unsigned char>(workDims[0], workDims[1], workDims[2], imageData);

	//Compute color histogram of image i.
	Histogram* histogram_i = initHistogram();
	computeRGBHistogram(image_i, histogram_i);

	//All-Gather each histogram to each other rank.
	double** histograms_data = new double*[N];

	MPI_Allgather(histogram_i->channels, 3 * 256, MPI_DOUBLE, histograms, N * 3 * 256, MPI_DOUBLE, MPI_COMM_WORLD);

	//imageResults should now be an array of 3x256 Double arrays. Proceed to similarity vector computation.
	similarities_i = computeSimilarityVector(histogram_i, histograms, N);

	//Match Rank 0's call to gather similarity vector results at Rank 0.
	std::cout << "\nSending data back to rank 0..."; //DEBUG
	MPI_Gather(similarities_i, N, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//Clean up
	delete[] similarities_i;
	for (int i=0 ; i < N ; i++){
		for (int j=0 ; j < 3 ; j++)
			delete[] histograms[i].channels[j];
	}
	delete[] histograms;

}

//N = communicatorSize
void do_rank_0_work(int N, char** imageNames)
{
	std::cout << "\nRank 0 reading image data for " << N << " images...";

	//Initialize Array of images as Packed3DArrays.
	cryph::Packed3DArray<unsigned char>** images = new cryph::Packed3DArray<unsigned char>*[N];
	for (int i=0 ; i < N ; i++) images[i] = new cryph::Packed3DArray<unsigned char>[N];


	//Initialize similarityResults array. NxN matrix of doubles.
	//Holds the results of each rank at the end of the computation
	double** similarityResults = new double*[N];
	for (int i=0 ; i < N ; i++) similarityResults[i] = new double[N];

	//Initialize Histogram results arrays. Nx3x256 doubles array.
	//Used to store rank i's 3x256 histogram results in histograms[i] (= 3x256 doubles array)
	Histogram* histograms = new Histogram[N];
	for (int img=0 ; img < N ; img++) {
		histograms[img].channels = new double*[3];
		for (int rgb=0 ; rgb < 3 ; rgb++) histograms[img].channels[rgb] = new double[256];
	}

	//Read N images into the array of images as Packed3DArrays.
	for (int img=0 ; img < N ; img++)
		if (ImageReader* ir = ImageReader::create(imageNames[img + 1]))
			images[img] = ir->getInternalPacked3DArrayImage();
		else return;

	//Distribute work. Work size message is blocking because its negligible size allows laziness.
	//Image sends are immediate, with a waitall after all requests are sent.
	std::cout << "\nRank 0 Distributing work sizes...";
	MPI_Request workSizeReq[N-1];
	MPI_Status workSizeStatus[N-1];
	MPI_Request imageSendReq[N-1];
	MPI_Status imageSendStatus[N-1];
	for (int receiver = 1; receiver < N; receiver++) {
		int workDims[3] = {images[receiver]->getDim1(), images[receiver]->getDim2(), images[receiver]->getDim3()};
		std::cout << "\nSending work size to rank " << receiver << ".";
		MPI_Isend(workDims, 3, MPI_INT, receiver, 0, MPI_COMM_WORLD, &workSizeReq[receiver-1]);	//tag: 0 - work size | 1 - work assignment (image data)
		std::cout << "\nSending image data to rank " << receiver << ".";
		MPI_Isend(images[receiver]->getData(), images[receiver]->getTotalNumberElements(), MPI_UNSIGNED_CHAR, receiver, 1, MPI_COMM_WORLD, &imageSendReq[receiver-1]);
	}
	MPI_Waitall(N-1, workSizeReq, workSizeStatus);
	MPI_Waitall(N-1, imageSendReq, imageSendStatus);
	std::cout << "\nAll ranks should have received work assignments.";

	// NOTE (Would use scatter to distribute array of images, but don't know buffer size
	//			required for each rank's work assignment to give the MPI call)
	// //Scatter images across each rank
	// std::cout << "\nRank 0 Scattering images...";
	// MPI_Request sendReq;
	// MPI_Iscatter(images, 1, MPI_P3DARRAY, MPI_IN_PLACE, 0, MPI_P3DARRAY, 0, MPI_COMM_WORLD, &sendReq);

	//[Rank 0 Work] Histogram work.
	Histogram* histogram_0 = initHistogram();
	computeRGBHistogram(images[0], histogram_0);
	histograms[0] = *histogram_0;

	//All-Gather each histogram to each other rank.
	std::cout << "\nAll-gathering computed color histograms...";
	MPI_Allgather(histogram_0->channels, 3 * 256, MPI_DOUBLE, histograms, N * 3 * 256, MPI_DOUBLE, MPI_COMM_WORLD);

	//[Rank 0 Work] Similarity vector work
	similarityResults[0] = computeSimilarityVector(histogram_0, histograms, N);

	//Gather each rank's similarity results together in Rank 0.
	std::cout << "\nGathering similarity results to rank 0...";
	MPI_Gather(MPI_IN_PLACE, 0, MPI_DOUBLE, similarityResults, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//Display output
	for (int r=0 ; r < N ; r++)
		for (int c=0 ; c < N ; c++)
			std::cout << ((!r) ? "\n": "") << similarityResults[r][c] << " ";

	//Clean up
	delete[] images;
	for (int i=0 ; i < N ; i++){
		for (int j=0 ; j < 3 ; j++)
			delete[] histograms[i].channels[j];
	}
	delete[] histograms;
	for (int i=0 ; i < N ; i++) delete[] similarityResults[i];
	delete[] similarityResults;
}


int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	//MPI setup
	int rank, communicatorSize;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); 						//Process UID 0 <= r < N
	MPI_Comm_size(MPI_COMM_WORLD, &communicatorSize); //N = communicatorSize

	if (argc < 2) { // (Abort if bad input)
		if (rank == 0) std::cerr << "Usage: " << argv[0] << " <image0> <...> <imageN>\n";
		return 0;
	}

	//Lazily ignore MPI command-line arguments
	int imageCount = 0;
	char** imageNames = new char*[imageCount];
	for (int i=1 ; i < argc ; i++)
		if (argv[i][0] != '-') {
			imageNames[imageCount] = argv[i];
			imageCount++;
		}

	//DEBUG
	std::cout << "\nFound " << imageCount << " images.";
	std::cout << "\nUsing communicator size " << communicatorSize << ".";

	//Make types known to MPI
	// mpiTypeP3DArray();
	// mpiTypeHistogram();

	//Entry point for work distribution.
	if (rank == 0) do_rank_0_work(communicatorSize, imageNames);
	else do_rank_i_work(communicatorSize);

	MPI_Finalize();
	return 0;
}


//NOTE Unused MPI Type definitions... relegating these to the bottom.
// //Packed 3D Array Type Definition
// void mpiTypeP3DArray()
// {
// 	//Packed 3D Arrays will be sent as a getDim
//
//
// 	//Block lengths, offsets, and types of members.
// 	int numFields = 1;
// 	int blklen[numFields] = {1, 1, 1};
// 	MPI_Aint displ[numFields] = {0, 0, 0,};
// 	MPI_Datatype types[numFields] = {MPI_INT, MPI_INT, MPI_INT};
//   //
// 	// //Compute offsets with a dummy P3DArray
// 	// cryph::Packed3DArray<unsigned char>* dummy;
//
// 	//From in-class reference
// 	// displ[0] = 0;
// 	// MPI_Aint base, oneField;
// 	// MPI_Get_address(&dummy.x, &base);
// 	// MPI_Get_address(&dummy.y, &oneField);
// 	// displ[1] = oneField - base; //Offset to y
// 	// MPI_Get_address(&dummy.z, &oneField);
// 	// displ[2] = oneField - base; //Offset to z
// 	// MPI_Get_address(&dummy.w, &oneField);
// 	// displ[3] = oneField - base; //Offset to w
//
// 	//Create with MPI and commit
// 	MPI_Type_create_struct(numFields, blklen, displ, types, MPI_P3DARRAY);
// 	MPI_Type_commit(MPI_P3DARRAY);
// }

// //Histogram Type Definition
// void mpiTypeHistogram()
// {
// 	//Block lengths, offsets, and types of members.
// 	int blklen[1] = {3 * 256};
// 	MPI_Aint displ[1] = {0};
// 	MPI_Datatype types[1] = {MPI_DOUBLE};
//
// 	//Create with MPI and commit
// 	MPI_Type_create_struct(1, blklen, displ, types, MPI_HISTOGRAM);
// 	MPI_Type_commit(MPI_HISTOGRAM);
// }
