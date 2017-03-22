/*
 * PageRankViennaCL.cu
 *
 *  Created on: Jun 30, 2015
 *      Author: yongchao
 */

/*ViennaCL*/
#include "PageRank.h"
struct viennaclOptions {
	viennaclOptions() {
		/*default settings*/
		_gpuId = 0;
		_nrepeats = 10;
		_sparseMatFormat = CSR_FORMAT;
		_doublePrecision = false;
	}

	/*variables*/
	int _gpuId;
	int _nrepeats;
	int _sparseMatFormat; /*sparse matrix format*/
	string _graphFileName; /*adjacency matrix of the graph stored in sparse matrix*/
	bool _doublePrecision; /*use single precision*/
	vector<int> _gpus;

	void printUsage() {
		cerr << "PageRankCuda viennacl [options]" << endl << "Options: " << endl
				<< "\t-i <str> (sparse matrix file name)" << endl
				<< "\t-d <int> (use double-precision floating point, default="
				<< _doublePrecision << ")" << endl
				<< "\t-r <int> (number of repeated runs, default=" << _nrepeats
				<< ")" << endl << "\t-g <int> (GPU index to use, default="
				<< _gpuId << ")" << endl << endl;
	}
	bool parseArgs(int argc, char* argv[]) {
		
		return true;
	}
};

int main_viennacl_pagerank(int argc, char* argv[]) {
	viennaclOptions options;

	/*parse parameters*/
	if (options.parseArgs(argc, argv) == false) {
		options.printUsage();
		return -1;
	}

	/*set the GPU device*/
	cudaSetDevice(options._gpus[options._gpuId]);
	CudaCheckError();

	/*perform SpMV*/
	bool ret = false;
	switch (options._sparseMatFormat) {
	case CSR_FORMAT:
		if (options._doublePrecision) {
			/*using double precision*/
			ret = pageRankViennaCL<unsigned int, double>(
					options._graphFileName.c_str(), options._nrepeats);

		} else {
			/*using single precision*/
			ret = pageRankViennaCL<unsigned int, float>(
					options._graphFileName.c_str(), options._nrepeats);
		}
		break;
	}
	return ret ? 0 : -1;
}

