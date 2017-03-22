/*
 * PageRankCuSparse.cu
 *
 *  Created on: May 29, 2015
 *      Author: Yongchao Liu
 *      Affiliation: School of Computational Science & Engineering, Georgia Institute of Technology
 *      Email: yliu@cc.gatech.edu
 */
#include "PageRank.h"

template<typename IntType, typename ValueType, typename CSRGraphType, typename VecType>
class cusparseCSRKernel {
public:
	cusparseCSRKernel() {
		_handle = 0;
		cusparseCreate(&_handle);
		CudaCheckError();

		_descr = 0;
		cusparseCreateMatDescr(&_descr);
		CudaCheckError();

		cusparseSetMatType(_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
		CudaCheckError();

		cusparseSetMatIndexBase(_descr, CUSPARSE_INDEX_BASE_ZERO);
		CudaCheckError();

		/*no transpose*/
		_op = CUSPARSE_OPERATION_NON_TRANSPOSE;
	}
	~cusparseCSRKernel() {
		/*release resources*/
		cusparseDestroyMatDescr(_descr);
		cusparseDestroy(_handle);
	}
	inline void convert(const CSRGraphType& graph, const ValueType dummy) {
		/*do nothing*/
	}
	inline void spmv(const CSRGraphType& graph, const VecType& x, VecType& y,
			const float alpha, const float beta) {
		cusparseScsrmv(_handle, _op, graph.num_rows, graph.num_cols,
				graph.num_entries, &alpha, _descr,
				thrust::raw_pointer_cast(graph.values.data()),
				thrust::raw_pointer_cast(graph.row_offsets.data()),
				thrust::raw_pointer_cast(graph.column_indices.data()),
				thrust::raw_pointer_cast(x.data()), &beta,
				thrust::raw_pointer_cast(y.data()));
	}
	inline void spmv(const CSRGraphType& graph, const VecType& x, VecType& y,
			const double alpha, const double beta) {
		/*invoke the kernel*/
		cusparseDcsrmv(_handle, _op, graph.num_rows, graph.num_cols,
				graph.num_entries, &alpha, _descr,
				thrust::raw_pointer_cast(graph.values.data()),
				thrust::raw_pointer_cast(graph.row_offsets.data()),
				thrust::raw_pointer_cast(graph.column_indices.data()),
				thrust::raw_pointer_cast(x.data()), &beta,
				thrust::raw_pointer_cast(y.data()));
	}
private:
	cusparseHandle_t _handle;
	cusparseOperation_t _op;
	cusparseMatDescr_t _descr;
};

/*HYB kernel*/
template<typename IntType, typename ValueType, typename CSRGraphType, typename VecType>
class cusparseHYBKernel {
public:
	cusparseHYBKernel() {
		_handle = 0;
		cusparseCreate(&_handle);
		CudaCheckError();

		_descr = 0;
		cusparseCreateMatDescr(&_descr);
		CudaCheckError();

		cusparseSetMatType(_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
		CudaCheckError();

		cusparseSetMatIndexBase(_descr, CUSPARSE_INDEX_BASE_ZERO);
		CudaCheckError();

		_op = CUSPARSE_OPERATION_NON_TRANSPOSE;

		/*create HYB format*/
		cusparseCreateHybMat(&_hybMat);
		CudaCheckError();
	}
	~cusparseHYBKernel() {
		/*release resources*/
		cusparseDestroyMatDescr(_descr);
		cusparseDestroy(_handle);
		cusparseDestroyHybMat(_hybMat);
	}

	/*sparse matrix format conversion*/
	void convert(const CSRGraphType& graph, float dummy) {

		/*convert from CSR to HYB*/
		cusparseScsr2hyb(_handle, graph.num_rows, graph.num_cols, _descr,
				thrust::raw_pointer_cast(graph.values.data()),
				thrust::raw_pointer_cast(graph.row_offsets.data()),
				thrust::raw_pointer_cast(graph.column_indices.data()), _hybMat,
				0, CUSPARSE_HYB_PARTITION_AUTO);
	}

	void convert(const CSRGraphType& graph, double dummy) {

		/*convert from CSR to HYB*/
		cusparseDcsr2hyb(_handle, graph.num_rows, graph.num_cols, _descr,
				thrust::raw_pointer_cast(graph.values.data()),
				thrust::raw_pointer_cast(graph.row_offsets.data()),
				thrust::raw_pointer_cast(graph.column_indices.data()), _hybMat,
				0, CUSPARSE_HYB_PARTITION_AUTO);
	}

	void spmv(const CSRGraphType& graph, const VecType& x, VecType& y,
			const float alpha, const float beta) {

		cusparseShybmv(_handle, _op, &alpha, _descr, _hybMat,
				thrust::raw_pointer_cast(x.data()), &beta,
				thrust::raw_pointer_cast(y.data()));
	}
	void spmv(const CSRGraphType& graph, const VecType& x, VecType& y,
			const double alpha, const double beta) {

		cusparseDhybmv(_handle, _op, &alpha, _descr, _hybMat,
				thrust::raw_pointer_cast(x.data()), &beta,
				thrust::raw_pointer_cast(y.data()));
	}
private:
	cusparseHandle_t _handle;
	cusparseOperation_t _op;
	cusparseMatDescr_t _descr;
	cusparseHybMat_t _hybMat;
};

struct cusparseOptions {
	cusparseOptions() {
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
		cerr << "PageRankCuda cusparse [options]" << endl << "Options: " << endl
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
		int c;

		/*GPU information*/
		int count;
		cudaDeviceProp prop;
		cudaGetDeviceCount(&count);
		CudaCheckError();

#if defined(HAVE_SM_35)
		cerr << "Require GPUs with compute capability >= 3.5" << endl;
#else
		cerr << "Require GPUs with compute capability >= 3.0" << endl;
#endif
		/*check the compute capability of GPUs*/
		for (int i = 0; i < count; ++i) {
			cudaGetDeviceProperties(&prop, i);
#if defined(HAVE_SM_35)
			if ((prop.major * 10 + prop.minor) >= 35) {
#else
			if ((prop.major * 10 + prop.minor) >= 30) {
#endif
				cerr << "GPU " << _gpus.size() << ": " << prop.name
					<< " (capability " << prop.major << "." << prop.minor
					<< ")" << endl;
				/*save the GPU*/
				_gpus.push_back(i);
			}
			}
		if (_gpus.size() == 0){
			cerr << "No qualified CUDA-enabled GPU is available" << endl;
			return false;
		}
		cerr << "Number of qualified GPUs: " << _gpus.size() << endl;

		_graphFileName = argv[1];
		//cin >> _graphFileName;
		_gpuId = 0;
		_nrepeats = 1;
		_doublePrecision = 0;
		return true;
	}
};

int main_cusparse_pagerank(int argc, char* argv[]) {
	cusparseOptions options;

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
					cusparseCSRKernel<int, double,
							cusp::csr_matrix<int, double, cusp::device_memory>,
							cusp::array1d<double, cusp::device_memory> > >(
					options._graphFileName.c_str(), options._nrepeats);

		} else {
			/*using single precision*/
			cout << "cang shu ok" << endl;
			ret = pageRank<int, float, cusp::device_memory,
					cusparseCSRKernel<int, float,
							cusp::csr_matrix<int, float, cusp::device_memory>,
							cusp::array1d<float, cusp::device_memory> > >(
					options._graphFileName.c_str(), options._nrepeats);
		}
		break;
	case HYB_FORMAT:
		if (options._doublePrecision) {
			/*using double precision*/
			ret = pageRank<int, double, cusp::host_memory,
					cusparseHYBKernel<int, double,
							cusp::csr_matrix<int, double, cusp::device_memory>,
							cusp::array1d<double, cusp::device_memory> > >(
					options._graphFileName.c_str(), options._nrepeats);

		} else {
			/*using single precision*/
			ret = pageRank<int, float, cusp::host_memory,
					cusparseHYBKernel<int, float,
							cusp::csr_matrix<int, float, cusp::device_memory>,
							cusp::array1d<float, cusp::device_memory> > >(
					options._graphFileName.c_str(), options._nrepeats);
		}
		break;
	}
	return ret ? 0 : -1;
}

