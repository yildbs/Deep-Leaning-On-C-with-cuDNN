#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"


#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}

#define checkCUDNN(status) {                                           \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure\nError: " << cudnnGetErrorString(status); \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure\nError: " << cudaGetErrorString(status); \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCublasErrors(status) {                                    \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cublas failure\nError code " << status;        \
      FatalError(_error.str());                                        \
    }                                                                  \
}


//TODO: memcpy from memory, not file
void readAllocMemcpy(const char* fname, int size, float*& data_h, float*& data_d)
{
    data_h = new float[size];

    readBinaryFile<value_type>(fname, size, *data_h);

    int size_b = size*sizeof(value_type);
    checkCudaErrors( cudaMalloc(data_d, size_b) );
    checkCudaErrors( cudaMemcpy(*data_d, *data_h,
                                size_b,
                                cudaMemcpyHostToDevice) );
}

struct Layer_t
{
    fp16Import_t fp16Import;
    int inputs;
    int outputs;
    // linear dimension (i.e. size is kernel_dim * kernel_dim)
    int kernel_dim;
    float *data_h, *data_d;
    float *bias_h, *bias_d;
    Layer_t()
    : data_h(NULL)
    , data_d(NULL)
    , bias_h(NULL)
    , bias_d(NULL)
    , inputs(0)
    , outputs(0)
    , kernel_dim(0)
    {};
    Layer_t( int _inputs
    		,int _outputs
    		,int _kernel_dim
    		,const char* fname_weights
    		,const char* fname_bias
    		,const char* pname = NULL
    		)
    : inputs(_inputs)
    , outputs(_outputs)
    , kernel_dim(_kernel_dim)
    {
        std::string weights_path, bias_path;
        if (pname != NULL)
        {
            get_path(weights_path, fname_weights, pname);
            get_path(bias_path, fname_bias, pname);
        }
        else
        {
            weights_path = fname_weights; bias_path = fname_bias;
        }
        readAllocInit(weights_path.c_str(), inputs * outputs * kernel_dim * kernel_dim,
                        &data_h, &data_d);
        readAllocInit(bias_path.c_str(), outputs, &bias_h, &bias_d);
    }
    ~Layer_t()
    {
        if (data_h != NULL) delete [] data_h;
        if (data_d != NULL) checkCudaErrors( cudaFree(data_d) );
        if (bias_h != NULL) delete [] bias_h;
        if (bias_d != NULL) checkCudaErrors( cudaFree(bias_d) );
    }
private:
    //TODO: memcpy from memory, not file
    void readAllocInit(const char* fname, int size, value_type** data_h, value_type** data_d)
    {
        readAllocMemcpy(fname, size, data_h, data_d);
    }
};


void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc,
                    cudnnTensorFormat_t& tensorFormat,
                    cudnnDataType_t& dataType,
                    int n,
                    int c,
                    int h,
                    int w)
{
    const int nDims = 4;
    int dimA[nDims] = {n,c,h,w};
    int strideA[nDims] = {c*h*w, h*w, w, 1};
    checkCUDNN( cudnnSetTensorNdDescriptor(tensorDesc,
                                            dataType,
                                            4,
                                            dimA,
                                            strideA ) );
}

class Network_t
{
	//typedef typename ScaleFactorTypeMap<value_type>::Type scaling_type;
	typedef float scaling_type;

    int convAlgorithm;
    cudnnDataType_t dataType;
    cudnnTensorFormat_t tensorFormat;
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnPoolingDescriptor_t     poolingDesc;
    cudnnActivationDescriptor_t  activDesc;
    cudnnLRNDescriptor_t   normDesc;
    cublasHandle_t cublasHandle;
    void createHandles()
    {
        checkCUDNN( cudnnCreate(&this->cudnnHandle) );
        checkCUDNN( cudnnCreateTensorDescriptor(&this->srcTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&this->dstTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&this->biasTensorDesc) );
        checkCUDNN( cudnnCreateFilterDescriptor(&this->filterDesc) );
        checkCUDNN( cudnnCreateConvolutionDescriptor(&this->convDesc) );
        checkCUDNN( cudnnCreatePoolingDescriptor(&this->poolingDesc) );
        checkCUDNN( cudnnCreateActivationDescriptor(&this->activDesc) );
        checkCUDNN( cudnnCreateLRNDescriptor(&this->normDesc) );

        checkCublasErrors( cublasCreate(&this->cublasHandle) );
    }
    void destroyHandles()
    {
    	checkCUDNN( cudnnDestroy(this->cudnnHandle) );

        checkCUDNN( cudnnDestroyTensorDescriptor(this->srcTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(this->dstTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(this->biasTensorDesc) );
        checkCUDNN( cudnnDestroyFilterDescriptor(this->filterDesc) );
        checkCUDNN( cudnnDestroyConvolutionDescriptor(this->convDesc) );
        checkCUDNN( cudnnDestroyPoolingDescriptor(this->poolingDesc) );
		checkCUDNN( cudnnDestroyActivationDescriptor(this->activDesc) );
        checkCUDNN( cudnnDestroyLRNDescriptor(this->normDesc) );

        checkCublasErrors( cublasDestroy(this->cublasHandle) );
    }
  public:
    Network_t()
    {
        convAlgorithm = -1;
        dataType = CUDNN_DATA_FLOAT;
        tensorFormat = CUDNN_TENSOR_NCHW;
        createHandles();
    };
    ~Network_t()
    {
        destroyHandles();
    }
    void cudaMemoryResize(int size, float*& data)
    {
        if (data != NULL)
        {
            checkCudaErrors( cudaFree(data) );
        }
        checkCudaErrors( cudaMalloc(&data, size*sizeof(float)) );
    }
    void setConvolutionAlgorithm(const cudnnConvolutionFwdAlgo_t& algo)
    {
        convAlgorithm = (int) algo;
    }
    void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, const Layer_t& layer, int c, float* data)
    {
        setTensorDesc(biasTensorDesc, tensorFormat, dataType, 1, c, 1, 1);

        float alpha = 1;
        float beta  = 1;
        checkCUDNN( cudnnAddTensor( cudnnHandle,
                                    &alpha,
                                    biasTensorDesc,
                                    layer.bias_d,
                                    &beta,
                                    dstTensorDesc,
                                    data) );
    }
    void FullyConnectedForward( const Layer_t& ip
    						  , int& n
    						  , int& c
    						  , int& h
    						  , int& w
    						  , float* srcData
    						  , float*& dstData)
    {
        if (n != 1)
        {
        	//TODO
            FatalError("Not Implemented");
        }
        int dim_x = c*h*w;
        int dim_y = ip.outputs;
        cudaMemoryResize(dim_y, dstData);

        float alpha = 1, beta = 1;
        // place bias into dstData
        checkCudaErrors( cudaMemcpy(dstData, ip.bias_d, dim_y*sizeof(float), cudaMemcpyDeviceToDevice) );

        gemv(cublasHandle, dim_x, dim_y, alpha,
                ip.data_d, srcData, beta,dstData);

        h = 1; w = 1; c = dim_y;
    }
    void ConvoluteForward(const Layer_t<value_type>& conv,
                          int& n, int& c, int& h, int& w,
                          value_type* srcData, value_type*& dstData)
    {
        cudnnConvolutionFwdAlgo_t algo;

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);

        const int tensorDims = 4;
        int tensorOuputDimA[tensorDims] = {n,c,h,w};
        const int filterDimA[tensorDims] = {conv.outputs, conv.inputs,
                                        conv.kernel_dim, conv.kernel_dim};

        checkCUDNN( cudnnSetFilterNdDescriptor(filterDesc,
                                              dataType,
                                              CUDNN_TENSOR_NCHW,
                                              tensorDims,
                                              filterDimA) );

        const int convDims = 2;
        int padA[convDims] = {0,0};
        int filterStrideA[convDims] = {1,1};
        int upscaleA[convDims] = {1,1};
        cudnnDataType_t  convDataType = dataType;
        if (dataType == CUDNN_DATA_HALF) {
            convDataType = CUDNN_DATA_FLOAT; //Math are done in FP32 when tensor are in FP16
        }
        checkCUDNN( cudnnSetConvolutionNdDescriptor(convDesc,
                                                    convDims,
                                                    padA,
                                                    filterStrideA,
                                                    upscaleA,
                                                    CUDNN_CROSS_CORRELATION,
                                                    convDataType) );
        // find dimension of convolution output
        checkCUDNN( cudnnGetConvolutionNdForwardOutputDim(convDesc,
                                                srcTensorDesc,
                                                filterDesc,
                                                tensorDims,
                                                tensorOuputDimA) );
        n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
        h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        if (convAlgorithm < 0)
        {
            // Choose the best according to the preference
            std::cout << "Testing cudnnGetConvolutionForwardAlgorithm ...\n";
            checkCUDNN( cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                    srcTensorDesc,
                                                    filterDesc,
                                                    convDesc,
                                                    dstTensorDesc,
                                                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                    0,
                                                    &algo
                                                    ) );
            std::cout << "Fastest algorithm is Algo " << algo << "\n";
            convAlgorithm = algo;
            // New way of finding the fastest config
            // Setup for findFastest call
            std::cout << "Testing cudnnFindConvolutionForwardAlgorithm ...\n";
            int requestedAlgoCount = 5;
            int returnedAlgoCount[1];
            cudnnConvolutionFwdAlgoPerf_t *results = (cudnnConvolutionFwdAlgoPerf_t*)malloc(sizeof(cudnnConvolutionFwdAlgoPerf_t)*requestedAlgoCount);
            checkCUDNN(cudnnFindConvolutionForwardAlgorithm( cudnnHandle,
                                                     srcTensorDesc,
                                                     filterDesc,
                                                     convDesc,
                                                     dstTensorDesc,
                                                     requestedAlgoCount,
                                                     returnedAlgoCount,
                                                     results
                                                   ) );
        for(int algoIndex = 0; algoIndex < *returnedAlgoCount; ++algoIndex){
            printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n", cudnnGetErrorString(results[algoIndex].status), results[algoIndex].algo, results[algoIndex].time, (unsigned long long)results[algoIndex].memory);
        }
            free(results);
        }
        else
        {
            algo = (cudnnConvolutionFwdAlgo_t)convAlgorithm;
            if (algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT)
            {
                //std::cout << "Using FFT for convolution\n";
            }
        }

        cudaMemoryResize(n*c*h*w, dstData);
        size_t sizeInBytes=0;
        void* workSpace=NULL;
        checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                srcTensorDesc,
                                                filterDesc,
                                                convDesc,
                                                dstTensorDesc,
                                                algo,
                                                &sizeInBytes) );
        if (sizeInBytes!=0)
        {
          checkCudaErrors( cudaMalloc(&workSpace,sizeInBytes) );
        }
        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnConvolutionForward(cudnnHandle,
                                              &alpha,
                                              srcTensorDesc,
                                              srcData,
                                              filterDesc,
                                              conv.data_d,
                                              convDesc,
                                              algo,
                                              workSpace,
                                              sizeInBytes,
                                              &beta,
                                              dstTensorDesc,
                                              dstData) );
        addBias(dstTensorDesc, conv, c, dstData);
        if (sizeInBytes!=0)
        {
          checkCudaErrors( cudaFree(workSpace) );
        }
    }

    void poolForward( int& n, int& c, int& h, int& w,
                      value_type* srcData, value_type** dstData)
    {
        const int poolDims = 2;
        int windowDimA[poolDims] = {2,2};
        int paddingA[poolDims] = {0,0};
        int strideA[poolDims] = {2,2};
        checkCUDNN( cudnnSetPoolingNdDescriptor(poolingDesc,
                                                CUDNN_POOLING_MAX,
                                                CUDNN_PROPAGATE_NAN,
                                                poolDims,
                                                windowDimA,
                                                paddingA,
                                                strideA ) );

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);

        const int tensorDims = 4;
        int tensorOuputDimA[tensorDims] = {n,c,h,w};
        checkCUDNN( cudnnGetPoolingNdForwardOutputDim(poolingDesc,
                                                    srcTensorDesc,
                                                    tensorDims,
                                                    tensorOuputDimA) );
        n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
        h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        cudaMemoryResize(n*c*h*w, dstData);
        scaling_type alpha = scaling_type(1);
        scaling_type beta = scaling_type(0);
        checkCUDNN( cudnnPoolingForward(cudnnHandle,
                                          poolingDesc,
                                          &alpha,
                                          srcTensorDesc,
                                          srcData,
                                          &beta,
                                          dstTensorDesc,
                                          *dstData) );
    }
    void softmaxForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        cudaMemoryResize(n*c*h*w, dstData);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnSoftmaxForward(cudnnHandle,
                                          CUDNN_SOFTMAX_ACCURATE ,
                                          CUDNN_SOFTMAX_MODE_CHANNEL,
                                          &alpha,
                                          srcTensorDesc,
                                          srcData,
                                          &beta,
                                          dstTensorDesc,
                                          *dstData) );
    }
    void lrnForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        unsigned lrnN = 5;
        double lrnAlpha, lrnBeta, lrnK;
        lrnAlpha = 0.0001; lrnBeta = 0.75; lrnK = 1.0;
        checkCUDNN( cudnnSetLRNDescriptor(normDesc,
                                            lrnN,
                                            lrnAlpha,
                                            lrnBeta,
                                            lrnK) );

        cudaMemoryResize(n*c*h*w, dstData);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnLRNCrossChannelForward(cudnnHandle,
                                            normDesc,
                                            CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData) );
    }
    void activationForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        checkCUDNN( cudnnSetActivationDescriptor(activDesc,
                                                CUDNN_ACTIVATION_RELU,
                                                CUDNN_PROPAGATE_NAN,
                                                0.0) );

        cudaMemoryResize(n*c*h*w, dstData);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnActivationForward(cudnnHandle,
                                            activDesc,
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData) );
    }

    int classify_example(const char* fname, const Layer_t<value_type>& conv1,
                          const Layer_t<value_type>& conv2,
                          const Layer_t<value_type>& ip1,
                          const Layer_t<value_type>& ip2)
    {
        int n,c,h,w;
        value_type *srcData = NULL, *dstData = NULL;
        value_type imgData_h[IMAGE_H*IMAGE_W];

        readImage(fname, imgData_h);

        std::cout << "Performing forward propagation ...\n";

        checkCudaErrors( cudaMalloc(&srcData, IMAGE_H*IMAGE_W*sizeof(value_type)) );
        checkCudaErrors( cudaMemcpy(srcData, imgData_h,
                                    IMAGE_H*IMAGE_W*sizeof(value_type),
                                    cudaMemcpyHostToDevice) );

        n = c = 1; h = IMAGE_H; w = IMAGE_W;
        ConvoluteForward(conv1, n, c, h, w, srcData, &dstData);
        poolForward(n, c, h, w, dstData, &srcData);

        ConvoluteForward(conv2, n, c, h, w, srcData, &dstData);
        poolForward(n, c, h, w, dstData, &srcData);

        FullyConnectedForward(ip1, n, c, h, w, srcData, &dstData);
        activationForward(n, c, h, w, dstData, &srcData);
        lrnForward(n, c, h, w, srcData, &dstData);

        FullyConnectedForward(ip2, n, c, h, w, dstData, &srcData);
        softmaxForward(n, c, h, w, srcData, &dstData);

        const int max_digits = 10;
        // Take care of half precision
        Convert<scaling_type> toReal;
        value_type result[max_digits];
        checkCudaErrors( cudaMemcpy(result, dstData, max_digits*sizeof(value_type), cudaMemcpyDeviceToHost) );
        int id = 0;
        for (int i = 1; i < max_digits; i++)
        {
            if (toReal(result[id]) < toReal(result[i])) id = i;
        }

        std::cout << "Resulting weights from Softmax:" << std::endl;
        printDeviceVector(n*c*h*w, dstData);

        checkCudaErrors( cudaFree(srcData) );
        checkCudaErrors( cudaFree(dstData) );
        return id;
    }
};


int main()
{
	std::string image_file_path = "/home/yildbs/Data/MNIST/test/test_000000.jpg";
	std::cout << "Hello !" << std::endl;


	cv::Mat img = cv::imread(image_file_path);
	cv::imshow("img", img);
	cv::waitKey(0);
}
