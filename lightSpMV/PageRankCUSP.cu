/*
 * PageRankCUSP.cu
 *
 *  Created on: May 29, 2015
 *      Author: Yongchao Liu
 *      Affiliation: School of Computational Science & Engineering, Georgia Institute of Technology
 *      Email: yliu@cc.gatech.edu
 */
#include "PageRank.h"
#include <cusp/detail/device/spmv/csr_vector.h>
#include <cusp/detail/device/spmv/hyb.h>

template<typename IntType, typename ValueType, typename CSRGraphType, typename VecType>
class cuspCSRKernel {
public:
	inline void convert(const CSRGraphType& graph, const ValueType dummy) {
		/*do nothing*/
	}
	inline void spmv(const CSRGraphType& graph, const VecType& x, VecType& y,
			const ValueType alpha, const ValueType beta) {

		/*invoke kernel*/
		cusp::detail::device::spmv_csr_vector_tex<CSRGraphType, VecType, VecType>(
				graph, x, y);

		/*synchronize*/
		cudaDeviceSynchronize();
	}
};

template<typename IntType, typename ValueType, typename CSRGraphType, typename VecType>
class cuspHYBKernel {
public:
	inline void convert(const CSRGraphType& graph, ValueType dummy) {
		cusp::convert(graph, _hybGraph);
	}
	inline void spmv(const CSRGraphType& graph, const VecType& x, VecType& y,
			const ValueType alpha, const ValueType beta) {

		/*invoke kernel*/
		cusp::detail::device::spmv_hyb_tex<
				cusp::hyb_matrix<IntType, ValueType, cusp::device_memory>, VecType>(_hybGraph,
				x, y);

		/*synchronize*/
		cudaDeviceSynchronize();
	}
private:
	cusp::hyb_matrix<IntType, ValueType, cusp::device_memory> _hybGraph;
};

struct cuspOptions {
	cuspOptions() {
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
		cerr << "PageRankCuda cusp [options]" << endl << "Options: " << endl
				<< "\t-i <str> (sparse matrix file name)" << endl
				<< "\t-f <int> (sparse matrix format, default="
				<< _sparseMatFormat << ")" << endl << "\t    0: CSR format"
				<< endl << "\t    1: HYB format" << endl
				<< "\t-d <int> (use double-precision floating point, default="
				<< _doublePrecision << ")" << endl
				<< "\t-r <int> (number of repeated runs, default=" << _nrepeats << ")" << endl
				<< "\t-g <int> (GPU index to use, default=" << _gpuId << ")"
				<< endl << endl;
	}
	bool parseArgs(int argc, char* argv[]) {
		return true;
	}
};

int main_cusp_pagerank(int argc, char* argv[]) {
	cuspOptions options;
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
			ret = pageRank<int, double, cusp::device_memory,
					cuspCSRKernel<int, double,
							cusp::csr_matrix<int, double, cusp::device_memory>,
							cusp::array1d<double, cusp::device_memory> > >(
					options._graphFileName.c_str(), options._nrepeats);

		} else {
			/*using single precision*/
			ret = pageRank<int, float, cusp::device_memory,
					cuspCSRKernel<int, float,
							cusp::csr_matrix<int, float, cusp::device_memory>,
							cusp::array1d<float, cusp::device_memory> > >(
					options._graphFileName.c_str(), options._nrepeats);
		}
		break;
	case HYB_FORMAT:
		if (options._doublePrecision) {
			/*using double precision*/
			ret = pageRank<int, double, cusp::host_memory,
					cuspHYBKernel<int, double,
							cusp::csr_matrix<int, double, cusp::device_memory>,
							cusp::array1d<double, cusp::device_memory> > >(
					options._graphFileName.c_str(), options._nrepeats);

		} else {
			/*using single precision*/
			ret = pageRank<int, float, cusp::host_memory,
					cuspHYBKernel<int, float,
							cusp::csr_matrix<int, float, cusp::device_memory>,
							cusp::array1d<float, cusp::device_memory> > >(
					options._graphFileName.c_str(), options._nrepeats);
		}
		break;
	}
	return ret ? 0 : -1;
}
