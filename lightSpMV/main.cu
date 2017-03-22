/*
 * main.cu
 *
 *  Created on: Nov 21, 2014
 *      Author: Yongchao Liu
 *      Affiliation: School of Computational Science & Engineering, Georgia Institute of Technology
 *      Email: yliu@cc.gatech.edu
 */
#include "PageRank.h"

static void printUsage()
{
	std::cerr << "PageRankCuda (v" << VERSION << ") is a set of PageRank implementations using different SpMV kernels using CUDA" << endl;
	std::cerr << "Usage: PageRankCuda cmd [options]" << endl
			<< "Commands:" << endl
			<< "    lightspmv    use the SpMV kernel of LightSpMV" << endl
			<< "    cusp         use the CSR-Vector kernel in Nvidia CUSP" << endl
			<< "    cusparse     use the CSR-based SpMV kernel in Nvidia CuSparse" << endl
			<< "    vienanacl    use the CSR-based SpMV kernel in ViennaCL" << endl
			<< endl;
}
template <typename ValueType>
 void test(int size)
{
	int maxK = 10000;
	cudaEvent_t start, end;
	float runtime;
	cusp::array1d<ValueType, cusp::host_memory> hostRank(size);
	cusp::array1d<ValueType, cusp::device_memory> devRank(size);


	/*create events*/
	cudaEventCreate(&start, cudaEventBlockingSync);
	cudaEventCreate(&end, cudaEventBlockingSync);

	/*starting measure the time*/
	cudaEventRecord(start, 0);

	for(int k = 0; k < maxK; ++k){
		thrust::copy(hostRank.begin(), hostRank.end(), devRank.begin());
		thrust::copy(devRank.begin(), devRank.end(), hostRank.begin());
	}

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&runtime, start, end);
	cerr << "Runtime of thrust::copy is " << runtime / maxK << endl;


	/*starting measure the time*/
	float runtime2;
	std::vector<ValueType> hostRank2(size);
	viennacl::vector<ValueType> devRank2(size);

	cudaEventRecord(start, 0);

	for(int k = 0; k < maxK; ++k){
		copy(hostRank2.begin(), hostRank2.end(), devRank2.begin());
		copy(devRank2.begin(), devRank2.end(), hostRank2.begin());
	}

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&runtime2, start, end);
	cerr << "Runtime of std::copy is " << runtime2 / maxK << endl;
	cerr << "Speeup of thrust::copy over std::copy is " << runtime2 / runtime << endl;
}
int32_t main(int32_t argc, char* argv[]) {
	char* cmd;
	int ret = -1;

	/*if(argc < 2){
		printUsage();
		return -1;
	}*/
	/*
#if 0
	//simple test
	cerr << "single precision test" <<endl;
	test<float>(1<<18);

	cerr << "double precison test" << endl;
	test<double>(1<<18);
#else*/
	/*test the command*/
	cmd = argv[1];
	if (cmd[0]=='l')
	{
		ret = main_lightspmv_pagerank(argc - 1, argv + 1);
	}
	else
	{
		ret = main_cusparse_pagerank(argc - 1, argv + 1);
	}
	return 0;
	
	/*//no use
	if(!strcmp(cmd, "lightspmv")){
		ret = main_lightspmv_pagerank(argc - 1, argv  + 1);
	}else if(!strcmp(cmd, "cusp")){
		ret = main_cusp_pagerank(argc - 1, argv + 1);
	}else if(!strcmp(cmd, "cusparse")){
		ret = main_cusparse_pagerank(argc - 1, argv + 1);
	}else if(!strcmp(cmd, "viennacl")){
		ret = main_viennacl_pagerank(argc - 1, argv + 1);
	}else{
		std::cerr << "Unsupported command: " << cmd << endl;
	}
#endif
	return ret;*/
}
