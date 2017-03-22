/*
 * PageRankLightSpMV.cu
 *
 *  Created on: May 29, 2015
 *      Author: Yongchao Liu
 *      Affiliation: School of Computational Science & Engineering, Georgia Institute of Technology
 *      Email: yliu@cc.gatech.edu
 */

#include "PageRank.h"
#include "LightSpMVCore.h"

/*formula Y = AX*/
template<typename IntType, typename ValueType, typename CSRGraphType,
		typename VecType>
class lightSpMVCSRKernel {
public:
	lightSpMVCSRKernel() {
		/*allocate space*/
		_cudaRowCounters.resize(1);

		/*specify the texture object parameters*/
		_texVectorX = 0;
		memset(&_texDesc, 0, sizeof(_texDesc));
		_texDesc.addressMode[0] = cudaAddressModeClamp;
		_texDesc.addressMode[1] = cudaAddressModeClamp;
		_texDesc.filterMode = cudaFilterModePoint;
		_texDesc.readMode = cudaReadModeElementType;

		/*clear*/
		memset(&_resDesc, 0, sizeof(_resDesc));

		/*get GPU information*/
		int device;
		cudaGetDevice(&device);
		CudaCheckError();

		/*get the device property*/
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, device);
		_numThreadsPerBlock = prop.maxThreadsPerBlock;
		_numThreadBlocks = prop.multiProcessorCount
				* (prop.maxThreadsPerMultiProcessor / _numThreadsPerBlock);
		//cerr << _numThreadsPerBlock << " " << _numThreadBlocks << endl;
	}
	inline void convert(const CSRGraphType& graph, const ValueType dummy) {
		/*do nothing*/
	}
	inline void spmv(const CSRGraphType& graph, const VecType& x, VecType& y,
			const float alpha, const float beta) {
		/*initialize the row counter*/
		_cudaRowCounters[0] = 0;

		/*texture object*/
		_resDesc.resType = cudaResourceTypeLinear;
		_resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0,
				cudaChannelFormatKindFloat);
		_resDesc.res.linear.devPtr = (void*) thrust::raw_pointer_cast(x.data());
		_resDesc.res.linear.sizeInBytes = x.size() * sizeof(float);
		cudaCreateTextureObject(&_texVectorX, &_resDesc, &_texDesc, NULL);
		CudaCheckError();

		int meanElementsPerRow = rint(
				(double) graph.num_entries / graph.num_rows);

		/*invoke the kernel*/
		if (meanElementsPerRow <= 2) {
			lightspmv::csr32DynamicWarp<float, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
					_numThreadBlocks, _numThreadsPerBlock>>>(
					thrust::raw_pointer_cast(_cudaRowCounters.data()),
					graph.num_rows, graph.num_cols,
					thrust::raw_pointer_cast(graph.row_offsets.data()),
					thrust::raw_pointer_cast(graph.column_indices.data()),
					thrust::raw_pointer_cast(graph.values.data()), _texVectorX,
					thrust::raw_pointer_cast(y.data()));
		} else if (meanElementsPerRow <= 4) {
			lightspmv::csr32DynamicWarp<float, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
					_numThreadBlocks, _numThreadsPerBlock>>>(
					thrust::raw_pointer_cast(_cudaRowCounters.data()),
					graph.num_rows, graph.num_cols,
					thrust::raw_pointer_cast(graph.row_offsets.data()),
					thrust::raw_pointer_cast(graph.column_indices.data()),
					thrust::raw_pointer_cast(graph.values.data()), _texVectorX,
					thrust::raw_pointer_cast(y.data()));
		} else if (meanElementsPerRow <= 64) {
			lightspmv::csr32DynamicWarp<float, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
					_numThreadBlocks, _numThreadsPerBlock>>>(
					thrust::raw_pointer_cast(_cudaRowCounters.data()),
					graph.num_rows, graph.num_cols,
					thrust::raw_pointer_cast(graph.row_offsets.data()),
					thrust::raw_pointer_cast(graph.column_indices.data()),
					thrust::raw_pointer_cast(graph.values.data()), _texVectorX,
					thrust::raw_pointer_cast(y.data()));

		} else {
			lightspmv::csr32DynamicWarp<float, 32,
					MAX_NUM_THREADS_PER_BLOCK / 32><<<_numThreadBlocks,
					_numThreadsPerBlock>>>(
					thrust::raw_pointer_cast(_cudaRowCounters.data()),
					graph.num_rows, graph.num_cols,
					thrust::raw_pointer_cast(graph.row_offsets.data()),
					thrust::raw_pointer_cast(graph.column_indices.data()),
					thrust::raw_pointer_cast(graph.values.data()), _texVectorX,
					thrust::raw_pointer_cast(y.data()));
		}
		/*synchronize*/
		cudaDeviceSynchronize();
	}
	inline void spmv(const CSRGraphType& graph, const VecType& x, VecType& y,
			const double alpha, const double beta) {
		/*initialize the row counter*/
		_cudaRowCounters[0] = 0;

		_resDesc.resType = cudaResourceTypeLinear;
		_resDesc.res.linear.desc = cudaCreateChannelDesc(32, 32, 0, 0,
				cudaChannelFormatKindSigned);
		_resDesc.res.linear.devPtr = (void*) thrust::raw_pointer_cast(x.data());
		_resDesc.res.linear.sizeInBytes = x.size() * sizeof(double);
		cudaCreateTextureObject(&_texVectorX, &_resDesc, &_texDesc, NULL);
		CudaCheckError();

		/*invoke the kernel*/
		int meanElementsPerRow = rint(
				(double) graph.num_entries / graph.num_rows);
		/*invoke the kernel*/
		if (meanElementsPerRow <= 2) {
			lightspmv::csr64DynamicWarp<double, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
					_numThreadBlocks, _numThreadsPerBlock>>>(
					thrust::raw_pointer_cast(_cudaRowCounters.data()),
					graph.num_rows, graph.num_cols,
					thrust::raw_pointer_cast(graph.row_offsets.data()),
					thrust::raw_pointer_cast(graph.column_indices.data()),
					thrust::raw_pointer_cast(graph.values.data()), _texVectorX,
					thrust::raw_pointer_cast(y.data()));

		} else if (meanElementsPerRow <= 4) {
			lightspmv::csr64DynamicWarp<double, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
					_numThreadBlocks, _numThreadsPerBlock>>>(
					thrust::raw_pointer_cast(_cudaRowCounters.data()),
					graph.num_rows, graph.num_cols,
					thrust::raw_pointer_cast(graph.row_offsets.data()),
					thrust::raw_pointer_cast(graph.column_indices.data()),
					thrust::raw_pointer_cast(graph.values.data()), _texVectorX,
					thrust::raw_pointer_cast(y.data()));
		} else if (meanElementsPerRow <= 64) {
			lightspmv::csr64DynamicWarp<double, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
					_numThreadBlocks, _numThreadsPerBlock>>>(
					thrust::raw_pointer_cast(_cudaRowCounters.data()),
					graph.num_rows, graph.num_cols,
					thrust::raw_pointer_cast(graph.row_offsets.data()),
					thrust::raw_pointer_cast(graph.column_indices.data()),
					thrust::raw_pointer_cast(graph.values.data()), _texVectorX,
					thrust::raw_pointer_cast(y.data()));
		} else {
			lightspmv::csr64DynamicWarp<double, 32,
					MAX_NUM_THREADS_PER_BLOCK / 32><<<_numThreadBlocks,
					_numThreadsPerBlock>>>(
					thrust::raw_pointer_cast(_cudaRowCounters.data()),
					graph.num_rows, graph.num_cols,
					thrust::raw_pointer_cast(graph.row_offsets.data()),
					thrust::raw_pointer_cast(graph.column_indices.data()),
					thrust::raw_pointer_cast(graph.values.data()), _texVectorX,
					thrust::raw_pointer_cast(y.data()));
		}
		/*synchronize*/
		cudaDeviceSynchronize();
	}
private:
	cusp::array1d<uint32_t, cusp::device_memory> _cudaRowCounters;

	/*for texture object*/
	cudaTextureDesc _texDesc;
	cudaResourceDesc _resDesc;
	cudaTextureObject_t _texVectorX;
	int _numThreadsPerBlock;
	int _numThreadBlocks;
};
template<typename IntType, typename ValueType, typename CSRGraphType,
		typename VecType>
class lightSpMVCSRKernelBLAS {
public:
	lightSpMVCSRKernelBLAS() {
		/*allocate space*/
		_cudaRowCounters.resize(1);

		/*specify the texture object parameters*/
		_texVectorX = 0;
		memset(&_texDesc, 0, sizeof(_texDesc));
		_texDesc.addressMode[0] = cudaAddressModeClamp;
		_texDesc.addressMode[1] = cudaAddressModeClamp;
		_texDesc.filterMode = cudaFilterModePoint;
		_texDesc.readMode = cudaReadModeElementType;

		/*clear*/
		memset(&_resDesc, 0, sizeof(_resDesc));

		/*get GPU information*/
		int device;
		cudaGetDevice(&device);
		CudaCheckError();

		/*get the device property*/
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, device);
		_numThreadsPerBlock = prop.maxThreadsPerBlock;
		_numThreadBlocks = prop.multiProcessorCount
				* (prop.maxThreadsPerMultiProcessor / _numThreadsPerBlock);
		//cerr << _numThreadsPerBlock << " " << _numThreadBlocks << endl;
	}
	inline void convert(const CSRGraphType& graph, const ValueType dummy) {
		/*do nothing*/
	}
	inline void spmv(const CSRGraphType& graph, const VecType& x, VecType& y,
			const float alpha, const float beta) {
		/*initialize the row counter*/
		_cudaRowCounters[0] = 0;

		/*texture object*/
		_resDesc.resType = cudaResourceTypeLinear;
		_resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0,
				cudaChannelFormatKindFloat);
		_resDesc.res.linear.devPtr = (void*) thrust::raw_pointer_cast(x.data());
		_resDesc.res.linear.sizeInBytes = x.size() * sizeof(float);
		cudaCreateTextureObject(&_texVectorX, &_resDesc, &_texDesc, NULL);
		CudaCheckError();

		int meanElementsPerRow = rint(
				(double) graph.num_entries / graph.num_rows);

		/*invoke the kernel*/
		if (meanElementsPerRow <= 2) {
			lightspmv::csr32DynamicWarpBLAS<float, 2,
					MAX_NUM_THREADS_PER_BLOCK / 2><<<_numThreadBlocks,
					_numThreadsPerBlock>>>(
					thrust::raw_pointer_cast(_cudaRowCounters.data()),
					graph.num_rows, graph.num_cols,
					thrust::raw_pointer_cast(graph.row_offsets.data()),
					thrust::raw_pointer_cast(graph.column_indices.data()),
					thrust::raw_pointer_cast(graph.values.data()), _texVectorX,
					thrust::raw_pointer_cast(y.data()), alpha, beta);
		} else if (meanElementsPerRow <= 4) {
			lightspmv::csr32DynamicWarpBLAS<float, 4,
					MAX_NUM_THREADS_PER_BLOCK / 4><<<_numThreadBlocks,
					_numThreadsPerBlock>>>(
					thrust::raw_pointer_cast(_cudaRowCounters.data()),
					graph.num_rows, graph.num_cols,
					thrust::raw_pointer_cast(graph.row_offsets.data()),
					thrust::raw_pointer_cast(graph.column_indices.data()),
					thrust::raw_pointer_cast(graph.values.data()), _texVectorX,
					thrust::raw_pointer_cast(y.data()), alpha, beta);
		} else if (meanElementsPerRow <= 64) {
			lightspmv::csr32DynamicWarpBLAS<float, 8,
					MAX_NUM_THREADS_PER_BLOCK / 8><<<_numThreadBlocks,
					_numThreadsPerBlock>>>(
					thrust::raw_pointer_cast(_cudaRowCounters.data()),
					graph.num_rows, graph.num_cols,
					thrust::raw_pointer_cast(graph.row_offsets.data()),
					thrust::raw_pointer_cast(graph.column_indices.data()),
					thrust::raw_pointer_cast(graph.values.data()), _texVectorX,
					thrust::raw_pointer_cast(y.data()), alpha, beta);

		} else {
			lightspmv::csr32DynamicWarpBLAS<float, 32,
					MAX_NUM_THREADS_PER_BLOCK / 32><<<_numThreadBlocks,
					_numThreadsPerBlock>>>(
					thrust::raw_pointer_cast(_cudaRowCounters.data()),
					graph.num_rows, graph.num_cols,
					thrust::raw_pointer_cast(graph.row_offsets.data()),
					thrust::raw_pointer_cast(graph.column_indices.data()),
					thrust::raw_pointer_cast(graph.values.data()), _texVectorX,
					thrust::raw_pointer_cast(y.data()), alpha, beta);
		}
		/*synchronize*/
		cudaDeviceSynchronize();
	}
	inline void spmv(const CSRGraphType& graph, const VecType& x, VecType& y,
			const double alpha, const double beta) {
		/*initialize the row counter*/
		_cudaRowCounters[0] = 0;

		_resDesc.resType = cudaResourceTypeLinear;
		_resDesc.res.linear.desc = cudaCreateChannelDesc(32, 32, 0, 0,
				cudaChannelFormatKindSigned);
		_resDesc.res.linear.devPtr = (void*) thrust::raw_pointer_cast(x.data());
		_resDesc.res.linear.sizeInBytes = x.size() * sizeof(double);
		cudaCreateTextureObject(&_texVectorX, &_resDesc, &_texDesc, NULL);
		CudaCheckError();

		/*invoke the kernel*/
		int meanElementsPerRow = rint(
				(double) graph.num_entries / graph.num_rows);
		/*invoke the kernel*/
		if (meanElementsPerRow <= 2) {
			lightspmv::csr64DynamicWarpBLAS<double, 2,
					MAX_NUM_THREADS_PER_BLOCK / 2><<<_numThreadBlocks,
					_numThreadsPerBlock>>>(
					thrust::raw_pointer_cast(_cudaRowCounters.data()),
					graph.num_rows, graph.num_cols,
					thrust::raw_pointer_cast(graph.row_offsets.data()),
					thrust::raw_pointer_cast(graph.column_indices.data()),
					thrust::raw_pointer_cast(graph.values.data()), _texVectorX,
					thrust::raw_pointer_cast(y.data()),
					thrust::raw_pointer_cast(y.data()), alpha, beta);

		} else if (meanElementsPerRow <= 4) {
			lightspmv::csr64DynamicWarpBLAS<double, 4,
					MAX_NUM_THREADS_PER_BLOCK / 4><<<_numThreadBlocks,
					_numThreadsPerBlock>>>(
					thrust::raw_pointer_cast(_cudaRowCounters.data()),
					graph.num_rows, graph.num_cols,
					thrust::raw_pointer_cast(graph.row_offsets.data()),
					thrust::raw_pointer_cast(graph.column_indices.data()),
					thrust::raw_pointer_cast(graph.values.data()), _texVectorX,
					thrust::raw_pointer_cast(y.data()),
					thrust::raw_pointer_cast(y.data()), alpha, beta);
		} else if (meanElementsPerRow <= 64) {
			lightspmv::csr64DynamicWarpBLAS<double, 8,
					MAX_NUM_THREADS_PER_BLOCK / 8><<<_numThreadBlocks,
					_numThreadsPerBlock>>>(
					thrust::raw_pointer_cast(_cudaRowCounters.data()),
					graph.num_rows, graph.num_cols,
					thrust::raw_pointer_cast(graph.row_offsets.data()),
					thrust::raw_pointer_cast(graph.column_indices.data()),
					thrust::raw_pointer_cast(graph.values.data()), _texVectorX,
					thrust::raw_pointer_cast(y.data()),
					thrust::raw_pointer_cast(y.data()), alpha, beta);
		} else {
			lightspmv::csr64DynamicWarpBLAS<double, 32,
					MAX_NUM_THREADS_PER_BLOCK / 32><<<_numThreadBlocks,
					_numThreadsPerBlock>>>(
					thrust::raw_pointer_cast(_cudaRowCounters.data()),
					graph.num_rows, graph.num_cols,
					thrust::raw_pointer_cast(graph.row_offsets.data()),
					thrust::raw_pointer_cast(graph.column_indices.data()),
					thrust::raw_pointer_cast(graph.values.data()), _texVectorX,
					thrust::raw_pointer_cast(y.data()),
					thrust::raw_pointer_cast(y.data()), alpha, beta);
		}
		/*synchronize*/
		cudaDeviceSynchronize();
	}
private:
	cusp::array1d<uint32_t, cusp::device_memory> _cudaRowCounters;

	/*for texture object*/
	cudaTextureDesc _texDesc;
	cudaResourceDesc _resDesc;
	cudaTextureObject_t _texVectorX;
	int _numThreadsPerBlock;
	int _numThreadBlocks;
};

struct lightSpmvOptions {
	lightSpmvOptions() {
		/*default settings*/
		_gpuId = 0;
		_kernel = 0;
		_nrepeats = 10;
		_doublePrecision = false;
	}

	/*variables*/
	int _gpuId;
	int _kernel;
	int _nrepeats;
	string _graphFileName; /*adjacency matrix of the graph stored in sparse matrix*/
	bool _doublePrecision; /*use single precision*/
	vector<int> _gpus;

	void printUsage() {
		cerr << "PageRankCuda lightspmv [options]" << endl << "Options: "
				<< endl << "\t-i <str> (sparse matrix file name)" << endl
				<< "\t-k <int> (which sparse matrix-vector multiplication kernel to use, default="
				<< _kernel << ")" << endl << "\t    0: formula Y = Ax" << endl
				<< "\t    1: formula Y = aAx + bY" << endl
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
		for (int i = 0; i < 1; ++i) {
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
		if (_gpus.size() == 0) {
			cerr << "No qualified CUDA-enabled GPU is available" << endl;
			return false;
		}
		cerr << "Number of qualified GPUs: " << _gpus.size() << endl;

		_graphFileName = argv[1];
		//cin >> _graphFileName;
		_kernel = 0;
		_gpuId = 0;
		_nrepeats = 1;
		_doublePrecision = 0;
		return true;
	}
		/*
		//parse parameters
		while ((c = getopt(argc, argv, "i:k:d:g:\n")) != -1) {
			switch (c) {
			case 'i':
				_graphFileName = optarg;
				break;
			case 'k':
				_kernel = atoi(optarg) ? 1 : 0;
				break;
			case 'g':
				_gpuId = atoi(optarg);
				break;
      case 'r':
        _nrepeats = atoi(optarg);
        if(_nrepeats < 1){
          _nrepeats = 1;
        }
        break;
			case 'd':
				_doublePrecision = atoi(optarg) == 0 ? false : true;
				break;
			default:
				cerr << "Unknown command: " << optarg << endl;
				return false;
			}
		}
		//check the file name
		if (_graphFileName.length() == 0) {
			cerr << "Graph must be given" << endl;
			return false;
		}

		//check GPU ID
		if (_gpuId >= (int) _gpus.size()) {
			_gpuId = _gpus.size() - 1;
		}
		if (_gpuId < 0) {
			_gpuId = 0;
		}
		return true;
	}*/
};

int main_lightspmv_pagerank(int argc, char* argv[]) {
	lightSpmvOptions options;
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
	switch (options._kernel) {
	case 0:
		if (options._doublePrecision) {
			/*using double precision*/
			ret = pageRank<uint32_t, double, cusp::device_memory,
					lightSpMVCSRKernel<uint32_t, double,
							cusp::csr_matrix<uint32_t, double,
									cusp::device_memory>,
							cusp::array1d<double, cusp::device_memory> > >(
					options._graphFileName.c_str(), options._nrepeats);

		} else {
			/*using single precision*/
			cout << "²ÎÊý¸ã¶¨" << endl;
			ret = pageRank<uint32_t, float, cusp::device_memory,
					lightSpMVCSRKernel<uint32_t, float,
							cusp::csr_matrix<uint32_t, float,
									cusp::device_memory>,
							cusp::array1d<float, cusp::device_memory> > >(
					options._graphFileName.c_str(), options._nrepeats);
		}
		break;
	case 1:
		if (options._doublePrecision) {
			/*using double precision*/
			ret = pageRank<uint32_t, double, cusp::device_memory,
					lightSpMVCSRKernelBLAS<uint32_t, double,
							cusp::csr_matrix<uint32_t, double,
									cusp::device_memory>,
							cusp::array1d<double, cusp::device_memory> > >(
					options._graphFileName.c_str(), options._nrepeats);

		} else {
			/*using single precision*/
			ret = pageRank<uint32_t, float, cusp::device_memory,
					lightSpMVCSRKernelBLAS<uint32_t, float,
							cusp::csr_matrix<uint32_t, float,
									cusp::device_memory>,
							cusp::array1d<float, cusp::device_memory> > >(
					options._graphFileName.c_str(), options._nrepeats);
		}
		break;
	}
	return ret ? 0 : -1;
}

